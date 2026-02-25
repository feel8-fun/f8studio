import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.nats_server_bootstrap import (  # noqa: E402
    _is_local_host,
    _parse_nats_host_port,
    _resolve_nats_server_binary,
    ensure_nats_server,
)


class NatsServerBootstrapTests(unittest.TestCase):
    def test_nats_bootstrap_url_parse_local_remote(self) -> None:
        host1, port1 = _parse_nats_host_port("nats://127.0.0.1:4222")
        self.assertEqual(host1, "127.0.0.1")
        self.assertEqual(port1, 4222)
        host2, port2 = _parse_nats_host_port("example.com:4333")
        self.assertEqual(host2, "example.com")
        self.assertEqual(port2, 4333)
        self.assertTrue(_is_local_host("localhost"))
        self.assertFalse(_is_local_host("example.com"))

    def test_nats_bootstrap_skip_remote_host(self) -> None:
        with patch("f8pysdk.nats_server_bootstrap._is_tcp_reachable", return_value=False), patch(
            "f8pysdk.nats_server_bootstrap._resolve_nats_server_binary"
        ) as resolve_mock:
            out = ensure_nats_server("nats://example.com:4222")
        self.assertFalse(out)
        resolve_mock.assert_not_called()

    def test_nats_bootstrap_uses_path_binary_first(self) -> None:
        fake = "/tmp/fake/nats-server"
        with patch("f8pysdk.nats_server_bootstrap.shutil.which", return_value=fake):
            resolved = _resolve_nats_server_binary(log_cb=None)
        self.assertEqual(resolved, Path(fake).resolve())

    def test_nats_bootstrap_download_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            expected = Path(td) / "installed" / "nats-server"
            expected.parent.mkdir(parents=True, exist_ok=True)
            with patch("f8pysdk.nats_server_bootstrap.shutil.which", return_value=None), patch(
                "f8pysdk.nats_server_bootstrap.Path.home", return_value=Path(td)
            ), patch(
                "f8pysdk.nats_server_bootstrap._download_latest_release_archive", return_value=Path(td) / "pkg.tar.gz"
            ) as dl_mock, patch(
                "f8pysdk.nats_server_bootstrap._extract_archive"
            ) as extract_mock, patch(
                "f8pysdk.nats_server_bootstrap._locate_binary", return_value=Path(td) / "extract" / "nats-server"
            ) as locate_mock, patch(
                "f8pysdk.nats_server_bootstrap._install_downloaded_binary", return_value=expected
            ) as install_mock:
                out = _resolve_nats_server_binary(log_cb=None)
        self.assertEqual(out, expected)
        dl_mock.assert_called_once()
        extract_mock.assert_called_once()
        locate_mock.assert_called_once()
        install_mock.assert_called_once()

    def test_nats_bootstrap_returns_false_on_download_failure(self) -> None:
        with patch("f8pysdk.nats_server_bootstrap._is_tcp_reachable", return_value=False), patch(
            "f8pysdk.nats_server_bootstrap._resolve_nats_server_binary", side_effect=RuntimeError("boom")
        ):
            out = ensure_nats_server("nats://127.0.0.1:4222")
        self.assertFalse(out)


if __name__ == "__main__":
    unittest.main()
