import io
import unittest
from contextlib import redirect_stderr

from geoxplain_aurora_adapter.remote.server_log import report_job_failure


class ServerErrorLoggingTests(unittest.TestCase):
    def test_failure_is_printed_once_with_log_context(self):
        job_id = "server-log-test-job"
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            first = report_job_failure(
                job_id,
                "requested timestamp unavailable",
                log_tail="Traceback details",
                source="TestBackend",
            )
            second = report_job_failure(job_id, "requested timestamp unavailable")

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(stderr.getvalue().count("requested timestamp unavailable"), 1)
        self.assertIn("[TestBackend]", stderr.getvalue())
        self.assertIn(job_id, stderr.getvalue())
        self.assertIn("Traceback details", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
