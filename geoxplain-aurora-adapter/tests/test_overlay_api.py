import io
import sys
import tempfile
import time
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np

import geoxplain_aurora_adapter as ax
from geoxplain_aurora_adapter.api import dispatch, timeparse
from geoxplain_aurora_adapter.schema.overlay import OverlayFrame, OverlayResult
from geoxplain_aurora_adapter.remote.local_overlay import LocalOverlayRunner
from geoxplain_aurora_adapter.remote.protocol import OverlayRequest


class OverlayApiTests(unittest.TestCase):
    def _overlay(self) -> OverlayResult:
        return OverlayResult(
            variable="q",
            level=850,
            label="Specific Humidity 850 hPa",
            unit="kg/kg",
            colormap="viridis",
            visible=False,
            lat=np.array([90.0, 89.75], dtype=np.float32),
            lon=np.array([0.0, 0.25, 0.5], dtype=np.float32),
            frames=[
                OverlayFrame(
                    timestamp="2020-01-01T00:00:00Z",
                    data=np.full((2, 3), 1.5, dtype=np.float32),
                ),
                OverlayFrame(
                    timestamp="2020-01-01T06:00:00Z",
                    data=np.full((2, 3), 2.5, dtype=np.float32),
                ),
            ],
        )

    def test_overlay_result_save_load_round_trip(self):
        result = self._overlay()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "humidity.overlay.npz"
            result.save(path)

            restored = OverlayResult.load(path)

        self.assertEqual(restored.variable, "q")
        self.assertEqual(restored.level, 850)
        self.assertEqual(restored.timestamps, [
            "2020-01-01T00:00:00Z",
            "2020-01-01T06:00:00Z",
        ])
        self.assertFalse(restored.visible)
        np.testing.assert_allclose(restored.frames[1].data, np.full((2, 3), 2.5))
        np.testing.assert_allclose(restored.lat, np.array([90.0, 89.75], dtype=np.float32))

    def test_overlay_result_default_offset_is_zero(self):
        self.assertEqual(self._overlay().overlay_offset_hours, 0)

    def test_overlay_offset_hours_round_trips(self):
        result = self._overlay()
        result.overlay_offset_hours = -6
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "humidity.overlay.npz"
            result.save(path)
            restored = OverlayResult.load(path)
        self.assertEqual(restored.overlay_offset_hours, -6)

    def test_overlay_result_msgpack_round_trip(self):
        try:
            import msgpack  # noqa: F401
        except ImportError:
            self.skipTest("msgpack is not installed")
        result = self._overlay()

        restored = OverlayResult.from_msgpack(result.to_msgpack())

        self.assertEqual(restored.label, "Specific Humidity 850 hPa")
        self.assertEqual(restored.unit, "kg/kg")
        np.testing.assert_allclose(restored.frames[0].data, result.frames[0].data)

    def test_overlay_timestamp_expansion_accepts_dates_ranges_and_exact_times(self):
        self.assertEqual(
            timeparse._expand_overlay_timestamps("2020-01-01", step_hours=6),
            [
                "2020-01-01T00:00:00Z",
                "2020-01-01T06:00:00Z",
                "2020-01-01T12:00:00Z",
                "2020-01-01T18:00:00Z",
            ],
        )
        self.assertEqual(
            timeparse._expand_overlay_timestamps(
                ["2020-01-01T06:00:00Z", "2020-01-02..2020-01-02"],
                step_hours=12,
            ),
            [
                "2020-01-01T06:00:00Z",
                "2020-01-02T00:00:00Z",
                "2020-01-02T12:00:00Z",
            ],
        )

    def test_pull_overlay_routes_remote_request(self):
        sentinel = object()
        with patch.object(dispatch, "_pull_overlay_remote", return_value=sentinel) as pull_remote:
            result = ax.pull_overlay(
                "q",
                "2020-01-01",
                level=850,
                remote="http://localhost:8765",
                name="Humidity",
                visible=False,
            )

        self.assertIs(result, sentinel)
        variable, timestamps, remote = pull_remote.call_args.args
        self.assertEqual(variable, "q")
        self.assertEqual(len(timestamps), 4)
        self.assertEqual(remote, "http://localhost:8765")
        self.assertEqual(pull_remote.call_args.kwargs["level"], 850)
        self.assertEqual(pull_remote.call_args.kwargs["name"], "Humidity")
        self.assertFalse(pull_remote.call_args.kwargs["visible"])

    def test_pull_overlay_infers_metadata_when_omitted(self):
        with patch.object(dispatch, "_pull_overlay_remote", return_value=object()) as pull_remote:
            ax.pull_overlay("q", "2020-01-01T00:00:00Z", level=850,
                            remote="http://localhost:8765")

        kwargs = pull_remote.call_args.kwargs
        self.assertEqual(kwargs["name"], "Specific Humidity 850 hPa")
        self.assertEqual(kwargs["unit"], "kg/kg")
        self.assertEqual(kwargs["colormap"], "viridis")

    def test_pull_overlay_infers_temperature_colormap(self):
        with patch.object(dispatch, "_pull_overlay_remote", return_value=object()) as pull_remote:
            ax.pull_overlay("t", "2020-01-01T00:00:00Z", level=850,
                            remote="http://localhost:8765")

        self.assertEqual(pull_remote.call_args.kwargs["colormap"], "thermal")

    def test_pull_overlay_rejects_invalid_colormap(self):
        with self.assertRaises(ValueError):
            ax.pull_overlay("q", "2020-01-01T00:00:00Z", level=850,
                            colormap="rainbow", remote="http://localhost:8765")

    def test_pull_overlay_rejects_unknown_overlay_time(self):
        with self.assertRaises(ValueError):
            ax.pull_overlay("q", "2020-01-01T00:00:00Z", level=850,
                            overlay_time="middle", remote="http://localhost:8765")

    def _fake_remote_overlay(self, captured):
        def fake(variable, timestamps, remote, **kwargs):
            captured["fetch"] = list(timestamps)
            return OverlayResult(
                variable=variable,
                level=kwargs.get("level"),
                label="x",
                frames=[
                    OverlayFrame(timestamp=ts, data=np.zeros((1, 1), dtype=np.float32))
                    for ts in timestamps
                ],
            )
        return fake

    def test_pull_overlay_default_input_does_not_shift(self):
        # The default overlay_time="input" pulls the field at the requested
        # (displayed = t1) time with no shift.
        captured = {}
        with patch.object(dispatch, "_pull_overlay_remote",
                          side_effect=self._fake_remote_overlay(captured)):
            result = ax.pull_overlay("q", "2024-01-16T12:00:00Z", level=850,
                                     remote="http://localhost:8765")
        self.assertEqual(captured["fetch"], ["2024-01-16T12:00:00Z"])
        self.assertEqual(result.timestamps, ["2024-01-16T12:00:00Z"])
        self.assertEqual(result.overlay_offset_hours, 0)

    def test_pull_overlay_prior_time_fetches_earlier_but_keeps_display_label(self):
        captured = {}
        with patch.object(dispatch, "_pull_overlay_remote",
                          side_effect=self._fake_remote_overlay(captured)):
            result = ax.pull_overlay("q", "2024-01-16T12:00:00Z", level=850,
                                     remote="http://localhost:8765",
                                     overlay_time="prior")
        # Field is read 6h earlier (the prior input step t0) ...
        self.assertEqual(captured["fetch"], ["2024-01-16T06:00:00Z"])
        # ... but the frame keeps the displayed (requested) timestamp.
        self.assertEqual(result.timestamps, ["2024-01-16T12:00:00Z"])
        self.assertEqual(result.overlay_offset_hours, -6)

    def test_pull_overlay_predicted_time_offset_is_plus_six(self):
        captured = {}
        with patch.object(dispatch, "_pull_overlay_remote",
                          side_effect=self._fake_remote_overlay(captured)):
            result = ax.pull_overlay("q", "2024-01-16T12:00:00Z", level=850,
                                     remote="http://localhost:8765",
                                     overlay_time="predicted")
        # Field is read 6h later: the forecast valid time t2.
        self.assertEqual(captured["fetch"], ["2024-01-16T18:00:00Z"])
        self.assertEqual(result.timestamps, ["2024-01-16T12:00:00Z"])
        self.assertEqual(result.overlay_offset_hours, 6)

    def test_pull_overlay_infers_dates_from_session(self):
        original = list(dispatch._SESSION_TIMESTAMPS)
        self.addCleanup(lambda: dispatch._SESSION_TIMESTAMPS.__setitem__(slice(None), original))
        dispatch._SESSION_TIMESTAMPS[:] = ["2024-01-16T12:00:00Z", "2024-01-16T18:00:00Z"]

        with patch.object(dispatch, "_pull_overlay_remote", return_value=object()) as pull_remote:
            ax.pull_overlay("q", level=850, remote="http://localhost:8765")

        _, timestamps, _ = pull_remote.call_args.args
        self.assertEqual(
            timestamps, ["2024-01-16T12:00:00Z", "2024-01-16T18:00:00Z"]
        )

    def test_pull_overlay_without_dates_or_session_raises(self):
        original = list(dispatch._SESSION_TIMESTAMPS)
        self.addCleanup(lambda: dispatch._SESSION_TIMESTAMPS.__setitem__(slice(None), original))
        dispatch._SESSION_TIMESTAMPS[:] = []

        with self.assertRaises(ValueError):
            ax.pull_overlay("q", level=850, remote="http://localhost:8765")

    def test_overlay_request_round_trip(self):
        req = OverlayRequest(
            variable="q",
            level=850,
            timestamps=["2020-01-01T00:00:00Z"],
            options={"name": "Humidity"},
        )

        restored = OverlayRequest.from_dict(req.to_dict())

        self.assertEqual(restored.variable, "q")
        self.assertEqual(restored.level, 850)
        self.assertEqual(restored.timestamps, ["2020-01-01T00:00:00Z"])
        self.assertEqual(restored.options, {"name": "Humidity"})


class LocalOverlayRunnerTests(unittest.TestCase):
    """Overlay pulls run in-process on the login node, never on a GPU job."""

    def _stub_compute(self, impl):
        """Install a fake ``geoxplain_aurora_adapter.engine.overlay_compute`` exposing ``impl``.

        ``overlay_compute`` imports torch (via ``_common``) at module load, which
        isn't available in the client-only test env, so ``LocalOverlayRunner``'s
        lazy import is fed a stub module instead.
        """
        mod_name = "geoxplain_aurora_adapter.engine.overlay_compute"
        original = sys.modules.get(mod_name)
        stub = types.ModuleType(mod_name)
        stub._pull_overlay_local = impl
        sys.modules[mod_name] = stub

        def restore():
            if original is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = original

        self.addCleanup(restore)

    def _await(self, runner, job_id, timeout_s=5.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            status = runner.status(job_id)
            if status.status in ("done", "error"):
                return status
            time.sleep(0.02)
        self.fail(f"job {job_id} did not finish within {timeout_s}s")

    def test_overlay_computed_locally_and_result_delivered(self):
        calls = []

        class _FakeResult:
            # Avoid the optional msgpack dependency in the result path.
            def to_msgpack(self) -> bytes:
                return b"PACKED-OVERLAY"

            def summary(self) -> str:
                return "OverlayResult(fake)"

        def fake_pull(variable, timestamps, **kwargs):
            calls.append((variable, timestamps, kwargs))
            return _FakeResult()

        self._stub_compute(fake_pull)
        runner = LocalOverlayRunner()
        job_id = runner.submit(
            OverlayRequest(
                variable="q",
                timestamps=["2020-01-01T00:00:00Z"],
                level=850,
                options={"name": "Humidity"},
            )
        )

        status = self._await(runner, job_id)
        self.assertEqual(status.status, "done")
        self.assertEqual(status.progress, 1.0)
        self.assertEqual(status.eta_s, 0.0)
        self.assertTrue(runner.owns(job_id))
        self.assertFalse(runner.owns("not-a-job"))
        self.assertEqual(
            calls,
            [("q", ["2020-01-01T00:00:00Z"], {"level": 850, "name": "Humidity"})],
        )
        self.assertEqual(runner.get_result(job_id), b"PACKED-OVERLAY")

    def test_overlay_failure_reported_as_error(self):
        def boom(*args, **kwargs):
            raise RuntimeError("no data on login node")

        self._stub_compute(boom)
        runner = LocalOverlayRunner()
        server_errors = io.StringIO()
        with redirect_stderr(server_errors):
            job_id = runner.submit(OverlayRequest(variable="q", timestamps=["t"], options={}))
            status = self._await(runner, job_id)
        self.assertEqual(status.status, "error")
        self.assertIn("no data on login node", status.error_message)
        self.assertIn(f"Job {job_id} FAILED: no data on login node", server_errors.getvalue())
        self.assertIn("Traceback", server_errors.getvalue())

    def test_unknown_job_raises_keyerror(self):
        runner = LocalOverlayRunner()
        with self.assertRaises(KeyError):
            runner.status("missing")
        with self.assertRaises(KeyError):
            runner.get_result("missing")

    def test_purge_skips_jobs_with_an_inflight_transfer(self):
        from geoxplain_aurora_adapter.remote import transfer_guard

        runner = LocalOverlayRunner()
        runner._jobs["job-1"] = {"status": "done", "result": b"x", "done_at": 0.0}

        transfer_guard.pin("job-1")
        self.addCleanup(transfer_guard.unpin, "job-1")
        # done_at=0 is far past any retention window, but the pin protects it.
        self.assertEqual(runner.purge(retention_s=1.0), 0)
        self.assertIn("job-1", runner._jobs)

        transfer_guard.unpin("job-1")
        self.assertEqual(runner.purge(retention_s=1.0), 1)
        self.assertNotIn("job-1", runner._jobs)


if __name__ == "__main__":
    unittest.main()
