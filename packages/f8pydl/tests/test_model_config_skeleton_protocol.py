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


class ModelConfigSkeletonProtocolTests(unittest.TestCase):
    def _write_yaml(self, content: str) -> Path:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        p = Path(tempdir.name) / "model.yaml"
        p.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
        return p

    def test_f8onnx_model_reads_model_skeleton_protocol(self) -> None:
        p = self._write_yaml(
            """
            schemaVersion: f8onnxModel/1
            model:
              id: demo
              task: yolo_pose
              onnxPath: demo.onnx
              skeletonProtocol: coco17
            input:
              width: 640
              height: 640
            """
        )
        spec = load_model_spec(p)
        self.assertEqual(spec.skeleton_protocol, "coco17")

    def test_legacy_reads_camel_case_field(self) -> None:
        p = self._write_yaml(
            """
            name: demo
            task: yolo_pose
            model_path: demo.onnx
            input_width: 640
            input_height: 640
            skeletonProtocol: human36m_17
            """
        )
        spec = load_model_spec(p)
        self.assertEqual(spec.skeleton_protocol, "human36m_17")

    def test_legacy_reads_snake_case_field(self) -> None:
        p = self._write_yaml(
            """
            name: demo
            task: yolo_pose
            model_path: demo.onnx
            input_width: 640
            input_height: 640
            skeleton_protocol: my_custom_19
            """
        )
        spec = load_model_spec(p)
        self.assertEqual(spec.skeleton_protocol, "my_custom_19")

    def test_defaults_to_none_when_missing(self) -> None:
        p = self._write_yaml(
            """
            name: demo
            task: yolo_det
            model_path: demo.onnx
            input_width: 640
            input_height: 640
            """
        )
        spec = load_model_spec(p)
        self.assertEqual(spec.skeleton_protocol, "none")

    def test_blank_protocol_normalizes_to_none(self) -> None:
        p = self._write_yaml(
            """
            schemaVersion: f8onnxModel/1
            model:
              id: demo
              task: yolo_pose
              onnxPath: demo.onnx
              skeletonProtocol: "   "
            input:
              width: 640
              height: 640
            """
        )
        spec = load_model_spec(p)
        self.assertEqual(spec.skeleton_protocol, "none")


if __name__ == "__main__":
    unittest.main()
