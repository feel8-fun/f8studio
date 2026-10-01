import os
import sys
import unittest


PKG_PYDL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PKG_PYDL not in sys.path:
    sys.path.insert(0, PKG_PYDL)


from f8pydl.onnx_runtime import OnnxTemporalWaveRuntime  # noqa: E402


class TcnRuntimeShapeTests(unittest.TestCase):
    def test_extract_fixed_input_shape_static(self) -> None:
        shape = OnnxTemporalWaveRuntime._extract_fixed_input_shape([1, 10, 3, 224, 224])
        self.assertEqual(shape, (10, 3, 224, 224))

    def test_extract_fixed_input_shape_dynamic_returns_none(self) -> None:
        shape = OnnxTemporalWaveRuntime._extract_fixed_input_shape([1, "seq", 3, 224, 224])
        self.assertIsNone(shape)

    def test_extract_fixed_input_shape_invalid_rank_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = OnnxTemporalWaveRuntime._extract_fixed_input_shape([1, 3, 224, 224])

    def test_extract_output_length_for_scalar_batch_output(self) -> None:
        out_len = OnnxTemporalWaveRuntime._extract_output_length([1])
        self.assertEqual(out_len, 1)

    def test_extract_output_length_for_sequence_output(self) -> None:
        out_len = OnnxTemporalWaveRuntime._extract_output_length([1, 10])
        self.assertEqual(out_len, 10)

    def test_extract_output_length_unsupported_rank_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = OnnxTemporalWaveRuntime._extract_output_length([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
