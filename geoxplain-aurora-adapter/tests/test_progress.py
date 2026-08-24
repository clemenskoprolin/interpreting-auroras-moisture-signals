import tempfile
import unittest

from geoxplain_aurora_adapter.engine.progress import (
    ProgressReporter,
    read_status_file,
    should_show_progress,
    write_status_file,
)
from geoxplain_aurora_adapter.remote.progress_log import format_progress_log
from geoxplain_aurora_adapter.remote.progress_mirror import apply_snapshot
from geoxplain_aurora_adapter.remote.protocol import JobStatus, log_tail_for_status


class ProgressTests(unittest.TestCase):
    def test_progress_reporter_renders_text_bar_and_eta(self):
        snapshots = []
        reporter = ProgressReporter(
            "ig batch",
            total_units=4,
            total_frames=2,
            min_interval_s=0.0,
            warmup_units=1,
            status_callback=snapshots.append,
        )

        reporter.set_frame(2)
        reporter.advance(1, phase="ig", detail="step 1/4")

        latest = snapshots[-1]
        self.assertEqual(latest["progress"], 0.25)
        self.assertIsNotNone(latest["eta_s"])
        self.assertIn("ig batch [", latest["text_output"])
        self.assertIn("25%", latest["text_output"])
        self.assertIn("frame 2/2", latest["text_output"])
        self.assertIn("step 1/4", latest["text_output"])

    def test_progress_reporter_accepts_fractional_advance(self):
        # The saliency per-block bar advances by fractions of one frame-unit
        # (forward/backward hooks), so advance() must accept floats and render
        # a clean integer count.
        snapshots = []
        reporter = ProgressReporter(
            "saliency",
            total_units=1,
            min_interval_s=0.0,
            warmup_units=1,
            status_callback=snapshots.append,
        )

        for _ in range(4):
            reporter.advance(0.1, phase="saliency", detail="forward")

        latest = snapshots[-1]
        self.assertAlmostEqual(latest["progress"], 0.4)
        self.assertIn("40%", latest["text_output"])
        self.assertIn("0/1", latest["text_output"])  # integer-rendered count

    def test_unknown_progress_renders_as_empty_zero_percent_bar(self):
        snapshots = []
        reporter = ProgressReporter(
            "saliency",
            min_interval_s=0.0,
            status_callback=snapshots.append,
        )

        reporter.set_phase("starting")

        latest = snapshots[-1]["text_output"]
        self.assertIn("[------------------------]   0%", latest)
        self.assertNotIn("?", latest)
        self.assertNotIn("--%", latest)

    def test_should_show_progress_throttles_small_steps(self):
        # First update always shows; then nothing until +5% or 15s of quiet.
        self.assertTrue(should_show_progress(0.01, None, 0.0))
        self.assertFalse(should_show_progress(0.04, 0.01, 1.0))   # +3%, recent
        self.assertTrue(should_show_progress(0.06, 0.01, 1.0))    # +5%
        self.assertTrue(should_show_progress(0.02, 0.01, 16.0))   # quiet too long
        # Indeterminate progress relies purely on the time fallback.
        self.assertFalse(should_show_progress(None, 0.5, 1.0))
        self.assertTrue(should_show_progress(None, 0.5, 16.0))

    def test_reporter_coalesces_sub_step_advances(self):
        snapshots = []
        reporter = ProgressReporter(
            "ig",
            total_units=100,
            min_interval_s=0.0,
            warmup_units=1,
            status_callback=snapshots.append,
        )

        for _ in range(6):  # 1% .. 6%
            reporter.advance(1, phase="ig")

        # Only the first (1%) and the one that crosses +5% (6%) surface.
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0]["progress"], 0.01)
        self.assertEqual(snapshots[-1]["progress"], 0.06)

        # A forced update (e.g. finish) always emits regardless of step.
        reporter.finish()
        self.assertEqual(snapshots[-1]["progress"], 1.0)

    def test_apply_snapshot_caches_and_prints_only_on_change(self):
        job: dict = {}
        snap = {"progress": 0.5, "eta_s": 12.0, "text_output": "ig [###---]  50%"}

        # First time: caches fields and returns the bar to print.
        self.assertEqual(apply_snapshot(job, snap), "ig [###---]  50%")
        self.assertEqual(job["progress"], 0.5)
        self.assertEqual(job["eta_s"], 12.0)

        # Same text again: cached, but nothing new to print.
        self.assertIsNone(apply_snapshot(job, snap))

        # New text: prints again.
        snap2 = {"progress": 0.6, "eta_s": 9.0, "text_output": "ig [####--]  60%"}
        self.assertEqual(apply_snapshot(job, snap2), "ig [####--]  60%")

        # A snapshot without text updates the cache but prints nothing.
        self.assertIsNone(apply_snapshot(job, {"progress": 0.7, "eta_s": 5.0}))
        self.assertEqual(job["progress"], 0.7)

    def test_progress_status_file_round_trip(self):
        snapshot = {
            "progress": 0.5,
            "eta_s": 12.0,
            "text_output": "rise [############------------]  50%",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/status.json"
            write_status_file(path, snapshot)

            self.assertEqual(read_status_file(path), snapshot)

    def test_job_status_includes_text_output(self):
        status = JobStatus(
            job_id="job-1",
            status="running",
            progress=0.5,
            eta_s=12.0,
            text_output="ig [############------------]  50%",
        )

        self.assertEqual(status.to_dict()["text_output"], status.text_output)

    def test_failed_job_status_keeps_actionable_log_context(self):
        log = "coverage details: " + ("x" * 3000)

        self.assertEqual(log_tail_for_status(log, "error"), log)
        self.assertEqual(len(log_tail_for_status(log, "running")), 200)

    def test_sbatch_progress_log_omits_timestamp_and_shortens_job_id(self):
        line = format_progress_log(
            "d8e19f0b-665a-42e9-bab1-2b3cc1ee083d",
            "saliency [------------------------]   0%",
        )

        self.assertEqual(
            line,
            "[d8e1...] saliency [------------------------]   0%",
        )


if __name__ == "__main__":
    unittest.main()
