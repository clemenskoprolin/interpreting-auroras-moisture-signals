import unittest
from types import SimpleNamespace
from unittest.mock import patch

import geoxplain_aurora_adapter as ax
from geoxplain_aurora_adapter.api import dispatch
from geoxplain_aurora_adapter.remote.protocol import BatchRunRequest


class BatchApiTests(unittest.TestCase):
    def _target(self):
        return ax.Target.box(
            var="q",
            level=850,
            lat=46.25,
            lon=8.75,
            size=(1.5, 2.5),
            timestamp="2020-04-20T12:00:00Z",
        )

    def test_expand_timeframe_targets_uses_start_count_and_step(self):
        targets = dispatch._expand_timeframe_targets(
            self._target(),
            timeframes=4,
            step_hours=6,
        )

        self.assertEqual(
            [target.timestamp for target in targets],
            [
                "2020-04-20T12:00:00Z",
                "2020-04-20T18:00:00Z",
                "2020-04-21T00:00:00Z",
                "2020-04-21T06:00:00Z",
            ],
        )

    def test_run_saliency_routes_remote_batch_request_when_timeframes_gt_one(self):
        with patch.object(dispatch, "_run_remote_batch") as run_remote_batch:
            sentinel = object()
            run_remote_batch.return_value = sentinel

            result = ax.run_saliency(
                target=self._target(),
                input=["t", "q"],
                timeframes=2,
                remote="http://localhost:8765",
            )

        self.assertIs(result, sentinel)
        method, targets, input_vars, remote = run_remote_batch.call_args.args
        self.assertEqual(method, "saliency")
        self.assertEqual([target.timestamp for target in targets], [
            "2020-04-20T12:00:00Z",
            "2020-04-20T18:00:00Z",
        ])
        self.assertEqual(input_vars, ["t", "q"])
        self.assertEqual(remote, "http://localhost:8765")

    def test_run_saliency_routes_single_request_by_default(self):
        with patch.object(dispatch, "_run_remote") as run_remote:
            sentinel = object()
            run_remote.return_value = sentinel

            result = ax.run_saliency(
                target=self._target(),
                input=["t", "q"],
                remote="http://localhost:8765",
            )

        self.assertIs(result, sentinel)
        self.assertEqual(run_remote.call_args.args[:4], (
            "saliency",
            self._target(),
            ["t", "q"],
            "http://localhost:8765",
        ))

    def test_rollout_rejects_step_hours_by_signature(self):
        with self.assertRaises(TypeError):
            ax.run_rollout(
                target=self._target(),
                input=["t", "q"],
                method="saliency",
                timeframes=4,
                step_hours=12,
            )

    def test_rollout_routes_local_request(self):
        with patch.object(dispatch, "_run_local_dispatch") as run_local:
            sentinel = object()
            run_local.return_value = sentinel

            result = ax.run_rollout(
                target=self._target(),
                input=["t", "q"],
                method="saliency",
                timeframes=4,
            )

        self.assertIs(result, sentinel)
        self.assertEqual(run_local.call_args.args[:3], (
            "saliency",
            self._target(),
            ["t", "q"],
        ))
        self.assertEqual(run_local.call_args.kwargs["_rollout_timeframes"], 4)

    def test_rollout_routes_remote_request(self):
        with patch.object(dispatch, "_run_remote") as run_remote:
            sentinel = object()
            run_remote.return_value = sentinel

            result = ax.run_rollout(
                target=self._target(),
                input=["t", "q"],
                method="saliency",
                timeframes=2,
                remote="http://localhost:8765",
            )

        self.assertIs(result, sentinel)
        self.assertEqual(run_remote.call_args.args[:4], (
            "saliency",
            self._target(),
            ["t", "q"],
            "http://localhost:8765",
        ))
        self.assertEqual(run_remote.call_args.kwargs["_rollout_timeframes"], 2)

    def test_rollout_forwards_ig_options(self):
        with patch.object(dispatch, "_run_local_dispatch") as run_local:
            ax.run_rollout(
                target=self._target(),
                input=["t", "q"],
                method="ig",
                timeframes=3,
                n_steps=8,
                baseline_sigma_deg=1.25,
            )

        self.assertEqual(run_local.call_args.kwargs["_rollout_timeframes"], 3)
        self.assertEqual(run_local.call_args.kwargs["n_steps"], 8)
        self.assertEqual(run_local.call_args.kwargs["baseline_sigma_deg"], 1.25)

    def test_rollout_rejects_unwired_method(self):
        with self.assertRaisesRegex(NotImplementedError, "only implemented"):
            ax.run_rollout(
                target=self._target(),
                input=["t", "q"],
                method="rise",
                timeframes=4,
            )

    def test_rollout_validates_method(self):
        with self.assertRaisesRegex(ValueError, "Unknown rollout method"):
            ax.run_rollout(
                target=self._target(),
                input=["t", "q"],
                method="unknown",
                timeframes=4,
            )

    def test_session_records_valid_frame_timestamps_not_input(self):
        """Session timestamps come from the returned frames (valid times),
        so an auto-overlay lines up with what the viewer displays."""
        original = list(dispatch._SESSION_TIMESTAMPS)
        self.addCleanup(
            lambda: dispatch._SESSION_TIMESTAMPS.__setitem__(slice(None), original)
        )
        dispatch._SESSION_TIMESTAMPS[:] = []

        # The frame carries the requested timestamp (12:00); session
        # timestamps mirror it verbatim so an auto-overlay lines up.
        fake_result = SimpleNamespace(
            frames=[SimpleNamespace(timestamp="2020-04-20T12:00:00Z")]
        )
        with patch(
            "geoxplain_aurora_adapter.remote.client.run_remote",
            return_value=fake_result,
        ):
            ax.run_saliency(
                target=self._target(),
                input=["t", "q"],
                remote="http://localhost:8765",
            )

        self.assertEqual(dispatch.session_timestamps(), ["2020-04-20T12:00:00Z"])

    def test_batch_run_request_round_trip(self):
        req = BatchRunRequest(
            method="saliency",
            targets=[self._target().to_dict()],
            input_vars=["t", "q"],
            options={"example": 1},
        )

        restored = BatchRunRequest.from_dict(req.to_dict())

        self.assertEqual(restored.method, "saliency")
        self.assertEqual(restored.targets[0]["timestamp"], "2020-04-20T12:00:00Z")
        self.assertEqual(restored.input_vars, ["t", "q"])
        self.assertEqual(restored.options, {"example": 1})


if __name__ == "__main__":
    unittest.main()
