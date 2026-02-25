import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk import headless_runner  # noqa: E402


class HeadlessRunnerBootstrapTests(unittest.TestCase):
    def test_runner_bootstrap_enabled_by_default(self) -> None:
        with patch("f8pysdk.headless_runner._ensure_bootstrap_or_raise") as bootstrap_mock, patch(
            "f8pysdk.headless_runner._run_headless", return_value=0
        ) as run_mock:
            code = headless_runner.main(["--session", "dummy.json"])
        self.assertEqual(code, 0)
        bootstrap_mock.assert_called_once_with(nats_url="nats://127.0.0.1:4222", bootstrap=True)
        run_mock.assert_called_once()

    def test_runner_no_bootstrap_flag_skips_bootstrap(self) -> None:
        with patch("f8pysdk.headless_runner._ensure_bootstrap_or_raise") as bootstrap_mock, patch(
            "f8pysdk.headless_runner._run_headless", return_value=0
        ) as run_mock:
            code = headless_runner.main(["--session", "dummy.json", "--no-bootstrap"])
        self.assertEqual(code, 0)
        bootstrap_mock.assert_called_once_with(nats_url="nats://127.0.0.1:4222", bootstrap=False)
        run_mock.assert_called_once()

    def test_runner_bootstrap_failure_returns_exit_3(self) -> None:
        with patch(
            "f8pysdk.headless_runner._ensure_bootstrap_or_raise", side_effect=RuntimeError("bootstrap failed")
        ), patch("f8pysdk.headless_runner._run_headless", return_value=0):
            code = headless_runner.main(["--session", "dummy.json"])
        self.assertEqual(code, 3)


if __name__ == "__main__":
    unittest.main()
