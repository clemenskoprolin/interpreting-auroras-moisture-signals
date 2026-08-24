import io
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from geoxplain_aurora_adapter.serving.config import get_config_section, parse_retention, write_config
from geoxplain_aurora_adapter.remote.cli import build_parser
from geoxplain_aurora_adapter.serving.bind_security import warn_if_public_bind
from geoxplain_aurora_adapter.serving.runtime_config import prepare_listener_settings

try:  # backend modules import FastAPI (the [server] extra); skip if absent.
    import fastapi  # noqa: F401
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False


class RetentionParsingTests(unittest.TestCase):
    def test_never_and_blank_keep_forever(self):
        for value in ("never", "none", "", "0", "off", None):
            self.assertIsNone(parse_retention(value), value)

    def test_unit_suffixes(self):
        self.assertEqual(parse_retention("24h"), 24 * 3600)
        self.assertEqual(parse_retention("7d"), 7 * 86400)
        self.assertEqual(parse_retention("30d"), 30 * 86400)
        self.assertEqual(parse_retention("90m"), 90 * 60)
        self.assertEqual(parse_retention("45s"), 45)

    def test_bare_number_is_seconds(self):
        self.assertEqual(parse_retention("3600"), 3600)
        self.assertEqual(parse_retention(120), 120)

    def test_garbage_falls_back_to_forever(self):
        self.assertIsNone(parse_retention("soon"))
        self.assertIsNone(parse_retention("-5h"))


@unittest.skipUnless(HAS_SERVER, "requires the [server] extra (fastapi)")
class WalltimeParsingTests(unittest.TestCase):
    def setUp(self):
        from geoxplain_aurora_adapter.remote.slurm import _parse_walltime_s
        self.parse = _parse_walltime_s

    def test_hms(self):
        self.assertEqual(self.parse("00:30:00"), 1800)
        self.assertEqual(self.parse("01:00:00"), 3600)

    def test_mm_ss_and_bare_minutes(self):
        self.assertEqual(self.parse("30:00"), 1800)
        self.assertEqual(self.parse("30"), 1800)

    def test_days(self):
        self.assertEqual(self.parse("1-00:00:00"), 86400)
        self.assertEqual(self.parse("2-12"), 2 * 86400 + 12 * 3600)

    def test_unparseable(self):
        self.assertIsNone(self.parse(""))
        self.assertIsNone(self.parse("forever"))


class PublicBindWarningTests(unittest.TestCase):
    def test_loopback_is_silent(self):
        for host in ("127.0.0.1", "localhost", "::1"):
            buf = io.StringIO()
            with redirect_stderr(buf):
                warn_if_public_bind(host)
            self.assertEqual(buf.getvalue(), "", host)

    def test_public_bind_warns(self):
        buf = io.StringIO()
        with redirect_stderr(buf):
            warn_if_public_bind("0.0.0.0")
        self.assertIn("WARNING", buf.getvalue())
        self.assertIn("unauthenticated", buf.getvalue())


class RetentionConfigTests(unittest.TestCase):
    def _write_config(self, config_path: Path, *, retention=None):
        config = {
            "setup": {"deployment": "login-node", "mode": "sbatch-oneshot"},
            "network": {"host": "127.0.0.1", "port": 8765, "remote_url": "http://localhost:8765"},
            "sbatch": {
                "account": "project42",
                "partition": "normal",
                "time": "00:30:00",
                "venv": "~/venv-aurora-xai",
            },
            "data": {"weatherbench2_paths": ["/wb2/a.zarr"]},
        }
        if retention is not None:
            config["retention"] = retention
        write_config(config_path, config)

    def test_default_retention_resolves_as_never(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "listen.toml"
            self._write_config(config_path)
            args = build_parser().parse_args([
                "--yes",
                "--config", str(config_path),
            ])
            with redirect_stdout(io.StringIO()):
                settings = prepare_listener_settings(args, gpu=False, sbatch=True)
            self.assertEqual(settings.result_retention, "never")
            self.assertEqual(get_config_section("retention", config_path), {})

    def test_default_memory_retention_resolves_as_1h(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "listen.toml"
            self._write_config(config_path)
            args = build_parser().parse_args([
                "--yes",
                "--config", str(config_path),
            ])
            with redirect_stdout(io.StringIO()):
                settings = prepare_listener_settings(args, gpu=False, sbatch=True)
            self.assertEqual(settings.memory_retention, "1h")

    def test_cli_flags_override_retention(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "listen.toml"
            self._write_config(config_path)
            args = build_parser().parse_args([
                "--yes",
                "--config", str(config_path),
                "--result-retention", "7d",
                "--memory-retention", "30m",
            ])
            with redirect_stdout(io.StringIO()):
                settings = prepare_listener_settings(args, gpu=False, sbatch=True)
            self.assertEqual(settings.result_retention, "7d")
            self.assertEqual(settings.memory_retention, "30m")

    def test_default_bind_host_is_loopback(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "listen.toml"
            self._write_config(config_path)
            args = build_parser().parse_args([
                "--yes",
                "--config", str(config_path),
            ])
            with redirect_stdout(io.StringIO()):
                settings = prepare_listener_settings(args, gpu=False, sbatch=True)
            self.assertEqual(settings.host, "127.0.0.1")


@unittest.skipUnless(HAS_SERVER, "requires the [server] extra (fastapi)")
class InprocPurgeTests(unittest.TestCase):
    def test_purge_drops_old_completed_jobs(self):
        from geoxplain_aurora_adapter.remote.inproc import InprocBackend

        backend = InprocBackend(memory_retention_s=500)
        with backend._jobs_lock:
            backend._jobs["old"] = {"status": "done", "done_at": time.time() - 1000}
            backend._jobs["fresh"] = {"status": "done", "done_at": time.time()}
            backend._jobs["running"] = {"status": "running", "done_at": None}
        with redirect_stdout(io.StringIO()):
            backend._purge()
        self.assertNotIn("old", backend._jobs)
        self.assertIn("fresh", backend._jobs)
        self.assertIn("running", backend._jobs)

    def test_no_retention_keeps_everything(self):
        from geoxplain_aurora_adapter.remote.inproc import InprocBackend

        backend = InprocBackend(memory_retention_s=None)
        with backend._jobs_lock:
            backend._jobs["old"] = {"status": "done", "done_at": time.time() - 10**6}
        with redirect_stdout(io.StringIO()):
            backend._purge()
        self.assertIn("old", backend._jobs)


@unittest.skipUnless(HAS_SERVER, "requires the [server] extra (fastapi)")
class OneshotPurgeTests(unittest.TestCase):
    def _backend(self, td, **kw):
        from geoxplain_aurora_adapter.remote.sbatch_config import SbatchConfig
        from geoxplain_aurora_adapter.remote.sbatch_oneshot import SbatchOneshotBackend

        cfg = SbatchConfig(output_dir=td, overlay_on_login=False).resolve()
        return SbatchOneshotBackend(cfg, **kw), cfg

    def _make_result_dir(self, cfg, job_id, age_s):
        job_dir = os.path.join(cfg.output_dir, "xia_results", job_id)
        os.makedirs(job_dir, exist_ok=True)
        out_path = os.path.join(job_dir, "result.xia.npz")
        Path(out_path).write_bytes(b"x")
        old = time.time() - age_s
        os.utime(out_path, (old, old))
        os.utime(job_dir, (old, old))
        return job_dir, out_path

    def test_disk_scan_removes_old_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            backend, cfg = self._backend(td, result_retention_s=500)
            old_dir, _ = self._make_result_dir(cfg, "old", age_s=1000)
            fresh_dir, _ = self._make_result_dir(cfg, "fresh", age_s=0)
            with redirect_stdout(io.StringIO()):
                backend._purge()
            self.assertFalse(os.path.exists(old_dir))
            self.assertTrue(os.path.exists(fresh_dir))

    def test_disk_scan_skips_active_job_dir(self):
        with tempfile.TemporaryDirectory() as td:
            backend, cfg = self._backend(td, result_retention_s=500)
            active_dir, out_path = self._make_result_dir(cfg, "running", age_s=1000)
            with backend._jobs_lock:
                backend._jobs["running"] = {"status": "running", "out_path": out_path}
            with redirect_stdout(io.StringIO()):
                backend._purge()
            self.assertTrue(os.path.exists(active_dir))

    def test_memory_purge_independent_of_disk(self):
        # memory bounded (500s), disk never → in-memory record dropped, dir kept.
        with tempfile.TemporaryDirectory() as td:
            backend, cfg = self._backend(td, memory_retention_s=500, result_retention_s=None)
            job_dir, out_path = self._make_result_dir(cfg, "old", age_s=1000)
            with backend._jobs_lock:
                backend._jobs["old"] = {
                    "status": "done", "done_at": time.time() - 1000, "out_path": out_path,
                }
            with redirect_stdout(io.StringIO()):
                backend._purge()
            self.assertNotIn("old", backend._jobs)
            self.assertTrue(os.path.exists(job_dir))


class _FakeResp:
    def __init__(self, *, json=None, content=None, status=200):
        self._json = json
        self.content = content
        self._status = status

    def raise_for_status(self):
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")

    def json(self):
        return self._json


class _FakeClient:
    """Stand-in for httpx.Client that answers worker status/result GETs."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url):
        if url.endswith("/result"):
            return _FakeResp(content=b"RESULTBYTES")
        if "wj-done" in url:
            return _FakeResp(json={"status": "done"})
        if "wj-error" in url:
            return _FakeResp(json={
                "status": "error",
                "error_message": "worker could not load requested timestamp",
                "log_tail": "worker traceback details",
            })
        return _FakeResp(json={"status": "running"})


@unittest.skipUnless(HAS_SERVER, "requires the [server] extra (fastapi)")
class PersistentDrainTests(unittest.TestCase):
    def _backend(self, td):
        from geoxplain_aurora_adapter.remote.sbatch_config import SbatchConfig
        from geoxplain_aurora_adapter.remote.sbatch_persistent import SbatchPersistentBackend

        cfg = SbatchConfig(output_dir=td, time="00:30:00").resolve()
        return SbatchPersistentBackend(cfg)

    def test_pending_on_filters_by_worker_and_status(self):
        with tempfile.TemporaryDirectory() as td:
            b = self._backend(td)
            with b._jobs_lock:
                b._jobs.update({
                    "a": {"status": "running", "worker_url": "u1", "worker_job_id": "wa"},
                    "b": {"status": "done", "worker_url": "u1", "worker_job_id": "wb"},
                    "c": {"status": "running", "worker_url": "u2", "worker_job_id": "wc"},
                    "d": {"status": "running", "worker_url": "u1", "worker_job_id": None},
                })
            self.assertEqual(b._pending_on("u1"), [("a", "wa")])

    def test_status_short_circuits_on_stored_done(self):
        with tempfile.TemporaryDirectory() as td:
            b = self._backend(td)
            with b._jobs_lock:
                b._jobs["d"] = {
                    "status": "done", "worker_url": "http://dead:1", "worker_job_id": "x",
                    "result": b"cached",
                }
            js = b.status("d")  # must NOT hit the (dead) worker_url
            self.assertEqual(js.status, "done")
            self.assertEqual(b.get_result("d"), b"cached")

    def test_worker_error_is_logged_and_cached_once(self):
        with tempfile.TemporaryDirectory() as td:
            b = self._backend(td)
            with b._jobs_lock:
                b._jobs["proxy-error"] = {
                    "status": "running",
                    "worker_url": "http://worker:1",
                    "worker_job_id": "wj-error",
                    "result": None,
                }

            server_errors = io.StringIO()
            with patch("httpx.Client", _FakeClient), redirect_stderr(server_errors):
                first = b.status("proxy-error")
                second = b.status("proxy-error")

            self.assertEqual(first.status, "error")
            self.assertEqual(second.status, "error")
            self.assertEqual(second.error_message, "worker could not load requested timestamp")
            self.assertEqual(server_errors.getvalue().count("Job proxy-error FAILED"), 1)
            self.assertIn("worker traceback details", server_errors.getvalue())

    def test_submit_is_nonblocking_and_reports_starting(self):
        import threading
        from geoxplain_aurora_adapter.remote.protocol import RunRequest

        with tempfile.TemporaryDirectory() as td:
            b = self._backend(td)
            # Pin the worker "provisioning" open so the dispatch thread can't
            # progress; submit() must still return immediately.
            gate = threading.Event()

            def blocked_ensure():
                gate.wait(timeout=10)
                raise RuntimeError("worker never came up")

            b._ensure_worker = blocked_ensure
            req = RunRequest(method="saliency", target={}, input_vars=["q"])

            t0 = time.time()
            with redirect_stdout(io.StringIO()):
                job_id = b.submit(req)
            elapsed = time.time() - t0

            self.assertLess(elapsed, 1.0)  # did not block on the SLURM queue
            self.assertEqual(b.status(job_id).status, "starting")
            gate.set()  # let the dispatch thread unwind

    def test_shutdown_cancels_worker(self):
        from geoxplain_aurora_adapter.remote import sbatch_persistent as sp

        with tempfile.TemporaryDirectory() as td:
            b = self._backend(td)
            with b._worker_lock:
                b._worker_slurm_id = "12345"
                b._worker_url = "http://worker:1"
            cancelled = []
            with patch.object(sp, "_scancel", side_effect=cancelled.append):
                with redirect_stdout(io.StringIO()):
                    b.shutdown()
            self.assertEqual(cancelled, ["12345"])
            self.assertIsNone(b._worker_url)
            # Idempotent: a second shutdown is a no-op (nothing left to cancel).
            cancelled.clear()
            with patch.object(sp, "_scancel", side_effect=cancelled.append):
                with redirect_stdout(io.StringIO()):
                    b.shutdown()
            self.assertEqual(cancelled, [])

    def test_drain_caches_done_and_fails_stuck(self):
        from geoxplain_aurora_adapter.remote import sbatch_persistent as sp

        with tempfile.TemporaryDirectory() as td:
            b = self._backend(td)
            old = "http://old-worker:55555"
            with b._jobs_lock:
                b._jobs.update({
                    "j-done": {"status": "running", "worker_url": old,
                               "worker_job_id": "wj-done", "result": None},
                    "j-stuck": {"status": "running", "worker_url": old,
                                "worker_job_id": "wj-stuck", "result": None},
                    "j-other": {"status": "running", "worker_url": "http://other",
                                "worker_job_id": "wj-x", "result": None},
                })
            with patch("httpx.Client", _FakeClient), \
                    patch.object(sp, "_DRAIN_POLL_INTERVAL_S", 0.02):
                with redirect_stdout(io.StringIO()):
                    b._drain_worker(old, deadline=time.time() + 0.2)

            # Completed job: pulled back and cached, served from the listener now.
            self.assertEqual(b._jobs["j-done"]["status"], "done")
            self.assertEqual(b._jobs["j-done"]["result"], b"RESULTBYTES")
            # Unfinished job at deadline: failed with a clear message, not hung.
            self.assertEqual(b._jobs["j-stuck"]["status"], "error")
            self.assertIn("wall-time", b._jobs["j-stuck"]["error"])
            # Job on a different worker: untouched.
            self.assertEqual(b._jobs["j-other"]["status"], "running")


if __name__ == "__main__":
    unittest.main()
