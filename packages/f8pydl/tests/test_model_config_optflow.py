import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


PKG_PYDL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PKG_PYDL not in sys.path:
    sys.path.insert(0, PKG_PYDL)


from f8pydl.model_config import load_model_spec  # noqa: E402


class ModelConfigOptflowTests(unittest.TestCase):
    def _write_yaml(self, content: str) -> Path:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        p = Path(tempdir.name) / "model.yaml"
        p.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
        return p

    def test_f8onnx_model_parses_optflow_task_and_defaults(self) -> None:
        p = self._write_yaml(
            """
            schemaVersion: f8onnxModel/1
            model:
              id: nf
              task: optflow_neuflowv2
              onnxPath: nf.onnx
            input:
              width: 640
              height: 640
            """
        )
        spec = load_model_spec(p)
        self.assertEqual(spec.task, "optflow_neuflowv2")
        self.assertEqual(spec.flow_format, "flow2_f16")
        self.assertEqual(spec.flow_input_order, "prev_now")

    def test_f8onnx_model_parses_optflow_block(self) -> None:
        p = self._write_yaml(
            """
            schemaVersion: f8onnxModel/1
            model:
              id: nf
              task: optflow
              onnxPath: nf.onnx
            input:
              width: 640
              height: 640
            optflow:
              flowFormat: flow2_f16
              inputOrder: now_prev
            """
        )
        spec = load_model_spec(p)
        self.assertEqual(spec.task, "optflow_neuflowv2")
        self.assertEqual(spec.flow_format, "flow2_f16")
        self.assertEqual(spec.flow_input_order, "now_prev")

    def test_f8onnx_model_parses_onnx_url_and_sha(self) -> None:
        p = self._write_yaml(
            """
            schemaVersion: f8onnxModel/1
            model:
              id: nf
              task: optflow_neuflowv2
              onnxPath: nf.onnx
              onnxUrl: https://example.com/nf.onnx
              onnxSHA256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
            input:
              width: 640
              height: 640
            """
        )
        spec = load_model_spec(p)
        self.assertEqual(spec.onnx_url, "https://example.com/nf.onnx")
        self.assertEqual(spec.onnx_sha256, "a" * 64)

    def test_f8onnx_model_parses_legacy_model_url_and_sha_alias(self) -> None:
        p = self._write_yaml(
            """
            schemaVersion: f8onnxModel/1
            model:
              id: nf
              task: optflow_neuflowv2
              onnxPath: nf.onnx
              url: https://example.com/nf.onnx
              sha256: cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            input:
              width: 640
              height: 640
            """
        )
        spec = load_model_spec(p)
        self.assertEqual(spec.onnx_url, "https://example.com/nf.onnx")
        self.assertEqual(spec.onnx_sha256, "c" * 64)

    def test_legacy_parses_url_alias(self) -> None:
        p = self._write_yaml(
            """
            name: nf
            task: optflow_neuflowv2
            model_path: nf.onnx
            url: https://example.com/nf.onnx
            onnxSHA256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            input_width: 768
            input_height: 432
            """
        )
        spec = load_model_spec(p)
        self.assertEqual(spec.onnx_url, "https://example.com/nf.onnx")
        self.assertEqual(spec.onnx_sha256, "b" * 64)

    def test_rejects_invalid_onnx_sha256(self) -> None:
        p = self._write_yaml(
            """
            schemaVersion: f8onnxModel/1
            model:
              id: nf
              task: optflow_neuflowv2
              onnxPath: nf.onnx
              onnxSHA256: bad
            input:
              width: 640
              height: 640
            """
        )
        with self.assertRaises(ValueError):
            _ = load_model_spec(p)

    def test_optflow_rejects_invalid_flow_format(self) -> None:
        p = self._write_yaml(
            """
            schemaVersion: f8onnxModel/1
            model:
              id: nf
              task: optflow_neuflowv2
              onnxPath: nf.onnx
            input:
              width: 640
              height: 640
            optflow:
              flowFormat: flow_rgb8
            """
        )
        with self.assertRaises(ValueError):
            _ = load_model_spec(p)

    def test_optflow_rejects_invalid_input_order(self) -> None:
        p = self._write_yaml(
            """
            schemaVersion: f8onnxModel/1
            model:
              id: nf
              task: optical_flow
              onnxPath: nf.onnx
            input:
              width: 640
              height: 640
            optflow:
              inputOrder: invalid_order
            """
        )
        with self.assertRaises(ValueError):
            _ = load_model_spec(p)

    def test_optflow_rejects_non_onnx_path(self) -> None:
        p = self._write_yaml(
            """
            name: nf
            task: optflow_neuflowv2
            model_path: nf.engine
            input_width: 640
            input_height: 640
            """
        )
        with self.assertRaises(ValueError):
            _ = load_model_spec(p)


if __name__ == "__main__":
    unittest.main()
