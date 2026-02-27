import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path


PKG_PYDL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PKG_PYDL not in sys.path:
    sys.path.insert(0, PKG_PYDL)


from f8pydl.weights_downloader import ensure_onnx_file  # noqa: E402


class WeightsDownloaderTests(unittest.TestCase):
    def test_download_file_url_with_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.onnx"
            target = root / "target.onnx"
            payload = b"fake-onnx-content-123"
            source.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()

            ensure_onnx_file(
                onnx_path=target,
                onnx_url=source.resolve().as_uri(),
                onnx_sha256=digest,
                timeout_s=5.0,
            )

            self.assertTrue(target.exists())
            self.assertEqual(target.read_bytes(), payload)

            ensure_onnx_file(
                onnx_path=target,
                onnx_url=source.resolve().as_uri(),
                onnx_sha256=digest,
                timeout_s=5.0,
            )
            self.assertEqual(target.read_bytes(), payload)

    def test_download_sha256_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.onnx"
            target = root / "target.onnx"
            source.write_bytes(b"fake-onnx-content-456")
            wrong_digest = "0" * 64

            with self.assertRaises(ValueError):
                ensure_onnx_file(
                    onnx_path=target,
                    onnx_url=source.resolve().as_uri(),
                    onnx_sha256=wrong_digest,
                    timeout_s=5.0,
                )

            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
