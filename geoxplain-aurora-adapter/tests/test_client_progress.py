import types
import unittest
from unittest.mock import patch

from geoxplain_aurora_adapter.remote.client import run_remote, run_remote_batch
from geoxplain_aurora_adapter.schema.spec import Target


class _FakeResponse:
    def __init__(self, payload=None, *, content=b"", status_code=200, text="",
                 headers=None, chunks=None):
        self._payload = payload or {}
        self.content = content
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        # When streamed, deliver the body as these chunks (defaults to one).
        self._chunks = chunks if chunks is not None else [content]

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _FakeHTTPStatusError(self.text)

    # Streaming protocol used by client._fetch_result_bytes.
    def read(self):
        return self.content

    def iter_bytes(self):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeHTTPStatusError(Exception):
    pass


class _FakeRequestError(Exception):
    pass


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ClientPollingTests(unittest.TestCase):
    def _target(self):
        return Target.point(
            var="q",
            level=850,
            lat=46.2,
            lon=8.8,
            timestamp="2024-04-20T00:00:00Z",
        )

    def test_running_status_tightens_poll_interval_before_sleeping(self):
        statuses = [
            {"status": "queued_locally"},
            {"status": "queued"},
            {"status": "running"},
            {
                "status": "done",
                "progress": 1.0,
                "text_output": "saliency [########################] 100%",
                "log_tail": "saliency [########################] 100%\nworker saved",
                "result_url": "/jobs/job-1/result",
            },
        ]
        sleep_calls = []
        printed = []
        sentinel = types.SimpleNamespace(summary=lambda: "fake result")

        fake_httpx = self._fake_httpx(statuses)

        with patch.dict("sys.modules", {"httpx": fake_httpx}), \
                patch("geoxplain_aurora_adapter.remote.client.XiaResult.from_msgpack",
                      return_value=sentinel), \
                patch("time.sleep", side_effect=lambda value: sleep_calls.append(value)), \
                patch("builtins.print", side_effect=lambda *args, **_kw: printed.append(" ".join(map(str, args)))):
            result = run_remote(
                "saliency",
                self._target(),
                ["t", "q"],
                "http://listener:8765",
                timeout_s=60,
                poll_interval_s=2.0,
                poll_max_s=30.0,
            )

        self.assertIs(result, sentinel)
        self.assertEqual(sleep_calls, [3.0, 4.5, 1.0])
        self.assertNotIn("saliency [########################] 100%", "\n".join(printed))
        self.assertNotIn("worker saved", "\n".join(printed))

    def test_starting_status_waits_without_error(self):
        # Persistent mode reports "starting" while the GPU worker is provisioned.
        # The client must keep polling (not raise) and surface a friendly note.
        statuses = [
            {"status": "starting"},
            {"status": "starting"},
            {"status": "running"},
            {"status": "done", "progress": 1.0, "result_url": "/jobs/job-1/result"},
        ]
        printed = []
        sentinel = types.SimpleNamespace(summary=lambda: "fake result")
        fake_httpx = self._fake_httpx(statuses)

        with patch.dict("sys.modules", {"httpx": fake_httpx}), \
                patch("geoxplain_aurora_adapter.remote.client.XiaResult.from_msgpack",
                      return_value=sentinel), \
                patch("time.sleep", side_effect=lambda *_a: None), \
                patch("builtins.print", side_effect=lambda *args, **_kw: printed.append(" ".join(map(str, args)))):
            result = run_remote(
                "saliency",
                self._target(),
                ["t", "q"],
                "http://listener:8765",
                timeout_s=60,
            )

        self.assertIs(result, sentinel)
        self.assertIn("provisioning GPU worker", "\n".join(printed))

    def test_run_remote_timeout_tracks_progress_inactivity_not_total_elapsed(self):
        statuses = [
            {"status": "running", "progress": 0.1, "text_output": "saliency 10%"},
            {"status": "running", "progress": 0.1, "text_output": "saliency 10%"},
            {"status": "running", "progress": 0.2, "text_output": "saliency 20%"},
            {"status": "running", "progress": 0.2, "text_output": "saliency 20%"},
            {"status": "running", "progress": 0.3, "text_output": "saliency 30%"},
            {"status": "done", "progress": 1.0, "result_url": "/jobs/job-1/result"},
        ]
        clock = _Clock()
        sentinel = types.SimpleNamespace(summary=lambda: "fake result")
        fake_httpx = self._fake_httpx(statuses)

        with patch.dict("sys.modules", {"httpx": fake_httpx, "tqdm.auto": None}), \
                patch("geoxplain_aurora_adapter.remote.client.XiaResult.from_msgpack",
                      return_value=sentinel), \
                patch("geoxplain_aurora_adapter.remote.client.time.monotonic",
                      side_effect=clock.monotonic), \
                patch("geoxplain_aurora_adapter.remote.client.time.sleep",
                      side_effect=clock.sleep), \
                patch("builtins.print"):
            result = run_remote(
                "saliency",
                self._target(),
                ["t", "q"],
                "http://listener:8765",
                timeout_s=2.0,
                poll_interval_s=1.0,
                poll_max_s=1.0,
            )

        self.assertIs(result, sentinel)
        self.assertGreater(clock.now, 2.0)

    def test_run_remote_times_out_when_progress_does_not_change(self):
        statuses = [
            {"status": "running", "progress": 0.1, "text_output": "saliency 10%"},
            {"status": "running", "progress": 0.1, "text_output": "saliency 10%"},
            {"status": "running", "progress": 0.1, "text_output": "saliency 10%"},
            {"status": "running", "progress": 0.1, "text_output": "saliency 10%"},
        ]
        clock = _Clock()
        fake_httpx = self._fake_httpx(statuses)

        with patch.dict("sys.modules", {"httpx": fake_httpx, "tqdm.auto": None}), \
                patch("geoxplain_aurora_adapter.remote.client.time.monotonic",
                      side_effect=clock.monotonic), \
                patch("geoxplain_aurora_adapter.remote.client.time.sleep",
                      side_effect=clock.sleep), \
                patch("builtins.print"):
            with self.assertRaises(TimeoutError) as ctx:
                run_remote(
                    "saliency",
                    self._target(),
                    ["t", "q"],
                    "http://listener:8765",
                    timeout_s=2.0,
                    poll_interval_s=1.0,
                    poll_max_s=1.0,
                )

        message = str(ctx.exception)
        self.assertIn("made no progress for 2s", message)
        self.assertIn("Last status: running", message)
        self.assertIn("Last progress: 10%", message)

    def test_run_remote_poll_errors_do_not_reset_progress_timeout(self):
        statuses = [
            {"status": "running", "progress": 0.1, "text_output": "saliency 10%"},
            _FakeRequestError("temporary outage"),
            _FakeRequestError("temporary outage"),
            _FakeRequestError("temporary outage"),
            _FakeRequestError("temporary outage"),
        ]
        clock = _Clock()
        fake_httpx = self._fake_httpx(statuses)

        with patch.dict("sys.modules", {"httpx": fake_httpx, "tqdm.auto": None}), \
                patch("geoxplain_aurora_adapter.remote.client.time.monotonic",
                      side_effect=clock.monotonic), \
                patch("geoxplain_aurora_adapter.remote.client.time.sleep",
                      side_effect=clock.sleep), \
                patch("builtins.print"):
            with self.assertRaises(TimeoutError) as ctx:
                run_remote(
                    "saliency",
                    self._target(),
                    ["t", "q"],
                    "http://listener:8765",
                    timeout_s=2.0,
                    poll_interval_s=1.0,
                    poll_max_s=1.0,
                )

        self.assertIn("made no progress for 2s", str(ctx.exception))

    def test_run_remote_batch_uses_progress_inactivity_timeout(self):
        statuses = [
            {"status": "running", "progress": 0.1, "text_output": "batch 10%"},
            {"status": "running", "progress": 0.1, "text_output": "batch 10%"},
            {"status": "running", "progress": 0.2, "text_output": "batch 20%"},
            {"status": "done", "progress": 1.0, "result_url": "/jobs/job-1/result"},
        ]
        clock = _Clock()
        sentinel = types.SimpleNamespace(summary=lambda: "fake batch result")
        fake_httpx = self._fake_httpx(statuses)

        with patch.dict("sys.modules", {"httpx": fake_httpx, "tqdm.auto": None}), \
                patch("geoxplain_aurora_adapter.remote.client.XiaResult.from_msgpack",
                      return_value=sentinel), \
                patch("geoxplain_aurora_adapter.remote.client.time.monotonic",
                      side_effect=clock.monotonic), \
                patch("geoxplain_aurora_adapter.remote.client.time.sleep",
                      side_effect=clock.sleep), \
                patch("builtins.print"):
            result = run_remote_batch(
                "saliency",
                [self._target(), self._target()],
                ["t", "q"],
                "http://listener:8765",
                timeout_s=1.3,
                poll_interval_s=1.0,
                poll_max_s=1.0,
            )

        self.assertIs(result, sentinel)
        self.assertGreater(clock.now, 1.3)

    def test_run_remote_batch_times_out_when_progress_does_not_change(self):
        statuses = [
            {"status": "running", "progress": 0.1, "text_output": "batch 10%"},
            {"status": "running", "progress": 0.1, "text_output": "batch 10%"},
            {"status": "running", "progress": 0.1, "text_output": "batch 10%"},
        ]
        clock = _Clock()
        fake_httpx = self._fake_httpx(statuses)

        with patch.dict("sys.modules", {"httpx": fake_httpx, "tqdm.auto": None}), \
                patch("geoxplain_aurora_adapter.remote.client.time.monotonic",
                      side_effect=clock.monotonic), \
                patch("geoxplain_aurora_adapter.remote.client.time.sleep",
                      side_effect=clock.sleep), \
                patch("builtins.print"):
            with self.assertRaises(TimeoutError) as ctx:
                run_remote_batch(
                    "saliency",
                    [self._target(), self._target()],
                    ["t", "q"],
                    "http://listener:8765",
                    timeout_s=1.0,
                    poll_interval_s=1.0,
                    poll_max_s=1.0,
                )

        message = str(ctx.exception)
        self.assertIn("Remote batch job job-1 made no progress for 1s", message)
        self.assertIn("Last status: running", message)

    def test_slow_result_download_reports_progress(self):
        from geoxplain_aurora_adapter.remote import result_fetch

        # Two chunks; the monotonic clock jumps past the 15s report interval
        # between them, so exactly one progress line should be emitted.
        resp = _FakeResponse(
            content=b"ab",
            headers={"content-length": "40000000"},
            chunks=[b"a" * 12_000_000, b"b" * 28_000_000],
        )

        class _Client:
            def stream(self, method, url):
                return resp

        printed = []
        # start → next_report=15; chunk1 at 100s reports (and resets to 115s);
        # chunk2 at 110s is before the next interval, so it stays silent.
        clock = iter([0.0, 100.0, 110.0])
        with patch("time.monotonic", side_effect=lambda: next(clock)):
            out = result_fetch._fetch_result_bytes(
                _Client(), "http://x/result", "saliency",
                write=lambda line: printed.append(line),
            )

        self.assertEqual(len(out), 40_000_000)
        self.assertEqual(len(printed), 1)
        self.assertIn("transferring result — 12.0/40.0 MB", printed[0])

    def _fake_httpx(self, statuses):
        statuses = list(statuses)

        class FakeClient:
            def __init__(self, timeout):
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, json):
                return _FakeResponse({"job_id": "job-1"})

            def get(self, url):
                item = statuses.pop(0)
                if isinstance(item, Exception):
                    raise item
                return _FakeResponse(item)

            def stream(self, method, url):
                return _FakeResponse(content=b"packed")

        return types.SimpleNamespace(
            Client=FakeClient,
            HTTPStatusError=_FakeHTTPStatusError,
            RequestError=_FakeRequestError,
        )


if __name__ == "__main__":
    unittest.main()
