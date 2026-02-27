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


class ModelConfigTcnWaveTests(unittest.TestCase):
    def _write_yaml(self, content: str) -> Path:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        path = Path(tempdir.name) / "model.yaml"
        path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
        return path

    def test_f8onnx_model_parses_tcn_wave(self) -> None:
        path = self._write_yaml(
            """
            schemaVersion: f8onnxModel/1
            model:
              id: tcn_demo
              task: tcn_wave
              onnxPath: tcn_demo.onnx
            input:
              width: 224
              height: 224
            """
        )
        spec = load_model_spec(path)
        self.assertEqual(spec.task, "tcn_wave")
        self.assertEqual(spec.model_id, "tcn_demo")
        self.assertEqual(spec.input_width, 224)
        self.assertEqual(spec.input_height, 224)

    def test_temporal_block_is_ignored_for_model_loading(self) -> None:
        path = self._write_yaml(
            """
            schemaVersion: f8onnxModel/1
            model:
              id: tcn_demo
              task: tcn_wave
              onnxPath: tcn_demo.onnx
            input:
              width: 224
              height: 224
            temporal:
              outputScale: 8.5
              outputBias: -1.25
              aggregation: now_last
            """
        )
        spec = load_model_spec(path)
        self.assertEqual(spec.task, "tcn_wave")

    def test_legacy_temporal_fields_are_ignored(self) -> None:
        path = self._write_yaml(
            """
            name: tcn_demo
            task: tcn_wave
            model_path: tcn_demo.onnx
            input_width: 224
            input_height: 224
            temporal_output_scale: 7.0
            temporal_output_bias: 2.0
            temporal_aggregation: invalid
            """
        )
        spec = load_model_spec(path)
        self.assertEqual(spec.task, "tcn_wave")
        self.assertEqual(spec.model_id, "tcn_demo")


if __name__ == "__main__":
    unittest.main()
