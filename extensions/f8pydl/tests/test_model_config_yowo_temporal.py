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


class ModelConfigYowoTemporalTests(unittest.TestCase):
    def _write_yaml(self, content: str) -> Path:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        path = Path(tempdir.name) / "model.yaml"
        path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
        return path

    def test_f8onnx_model_parses_temporal_detector_block(self) -> None:
        path = self._write_yaml(
            """
            schemaVersion: f8onnxModel/1
            model:
              id: yowo_demo
              task: yowo_temporal_det
              onnxPath: yowo_demo.onnx
            input:
              width: 320
              height: 320
            temporal:
              clipLength: 16
              samplingRate: 2
              maxDet: 123
              resizeMode: direct_resize
              normalization: imagenet
            labels:
              classes:
                - insertive_actor
                - receptive_actor
            """
        )
        spec = load_model_spec(path)
        self.assertEqual(spec.task, "yowo_temporal_det")
        self.assertEqual(spec.temporal_clip_length, 16)
        self.assertEqual(spec.temporal_sampling_rate, 2)
        self.assertEqual(spec.temporal_max_det, 123)
        self.assertEqual(spec.temporal_resize_mode, "direct_resize")
        self.assertEqual(spec.temporal_normalization, "imagenet")

    def test_rejects_legacy_yowov3_temporal_alias(self) -> None:
        path = self._write_yaml(
            """
            type: yowov3_temporal
            """
        )
        with self.assertRaises(ValueError):
            _ = load_model_spec(path)

    def test_temporal_defaults_are_applied_when_block_is_omitted(self) -> None:
        path = self._write_yaml(
            """
            schemaVersion: f8onnxModel/1
            model:
              id: yowo_demo
              task: yowo_temporal_det
              onnxPath: yowo_demo.onnx
            input:
              width: 320
              height: 320
            """
        )
        spec = load_model_spec(path)
        self.assertEqual(spec.temporal_clip_length, 16)
        self.assertEqual(spec.temporal_sampling_rate, 1)
        self.assertEqual(spec.temporal_max_det, 300)
        self.assertEqual(spec.temporal_resize_mode, "direct_resize")
        self.assertEqual(spec.temporal_normalization, "imagenet")


if __name__ == "__main__":
    unittest.main()
