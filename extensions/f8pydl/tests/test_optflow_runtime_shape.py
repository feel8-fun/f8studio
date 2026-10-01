import os
import sys
import unittest


PKG_PYDL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PKG_PYDL not in sys.path:
    sys.path.insert(0, PKG_PYDL)


from f8pydl.onnx_runtime import OnnxNeuFlowRuntime  # noqa: E402


class OptflowRuntimeShapeTests(unittest.TestCase):
    def test_extract_fixed_hw_returns_dims_for_static_nchw(self) -> None:
        hw = OnnxNeuFlowRuntime._extract_fixed_hw([1, 3, 432, 768])
        self.assertEqual(hw, (432, 768))

    def test_extract_fixed_hw_returns_none_for_dynamic_dims(self) -> None:
        hw = OnnxNeuFlowRuntime._extract_fixed_hw([1, 3, "h", "w"])
        self.assertIsNone(hw)

    def test_extract_fixed_hw_returns_none_for_non_4d(self) -> None:
        hw = OnnxNeuFlowRuntime._extract_fixed_hw([1, 3, 432])
        self.assertIsNone(hw)


if __name__ == "__main__":
    unittest.main()
