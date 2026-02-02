# f8pydetect_tracker

Feel8 ONNX detector + tracker runtime service.

Notes:
- This repo currently uses CPython 3.14 in some environments; `onnxruntime` wheels may not exist for that version yet.
- For NVIDIA GPU acceleration, install `onnxruntime-gpu` (supported Python) and ensure `CUDAExecutionProvider` is available.
