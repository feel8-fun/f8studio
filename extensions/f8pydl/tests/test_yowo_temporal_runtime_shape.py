import os
import sys
import unittest

import numpy as np


PKG_PYDL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PKG_PYDL not in sys.path:
    sys.path.insert(0, PKG_PYDL)


from f8pydl.onnx_runtime import OnnxYowoTemporalDetectorRuntime  # noqa: E402


class YowoTemporalRuntimeShapeTests(unittest.TestCase):
    def test_extract_input_spec_bcthw(self) -> None:
        spec = OnnxYowoTemporalDetectorRuntime._extract_input_spec(
            [1, 3, 16, 320, 320],
            default_clip_length=8,
            default_height=224,
            default_width=224,
        )
        self.assertEqual(spec.layout, "bcthw")
        self.assertEqual(spec.clip_length, 16)
        self.assertEqual(spec.channels, 3)
        self.assertEqual(spec.input_height, 320)
        self.assertEqual(spec.input_width, 320)

    def test_extract_input_spec_btchw(self) -> None:
        spec = OnnxYowoTemporalDetectorRuntime._extract_input_spec(
            [1, 16, 3, 320, 320],
            default_clip_length=8,
            default_height=224,
            default_width=224,
        )
        self.assertEqual(spec.layout, "btchw")
        self.assertEqual(spec.clip_length, 16)
        self.assertEqual(spec.channels, 3)

    def test_extract_input_spec_uses_defaults_for_dynamic_dims(self) -> None:
        spec = OnnxYowoTemporalDetectorRuntime._extract_input_spec(
            [1, 3, "frames", "height", "width"],
            default_clip_length=16,
            default_height=320,
            default_width=320,
        )
        self.assertEqual(spec.layout, "bcthw")
        self.assertEqual(spec.clip_length, 16)
        self.assertEqual(spec.input_height, 320)
        self.assertEqual(spec.input_width, 320)

    def test_extract_input_spec_rejects_invalid_rank(self) -> None:
        with self.assertRaises(ValueError):
            _ = OnnxYowoTemporalDetectorRuntime._extract_input_spec(
                [1, 3, 320, 320],
                default_clip_length=16,
                default_height=320,
                default_width=320,
            )

    def test_map_boxes_to_frame_uses_direct_resize_scaling(self) -> None:
        boxes = np.asarray([[32.0, 16.0, 160.0, 80.0]], dtype=np.float32)
        mapped = OnnxYowoTemporalDetectorRuntime._map_boxes_to_frame(
            boxes,
            frame_size_hw=(320, 640),
            input_width=320,
            input_height=160,
        )
        self.assertEqual(mapped.shape, (1, 4))
        self.assertTrue(np.allclose(mapped[0], np.asarray([64.0, 32.0, 320.0, 160.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
