# f8dlinfer (experimental)

Experimental C++ inference service package for Phase B (TensorRT-oriented path).

Current scope:
- Provide a buildable service skeleton behind `F8_ENABLE_DLINFER`.
- Reserve service process shape and `--describe` behavior.

Next scope:
- Add TensorRT backend (`.engine` first, `.onnx` offline conversion path).
- Match `f8visionDetections/1` output exactly with Python baseline.
