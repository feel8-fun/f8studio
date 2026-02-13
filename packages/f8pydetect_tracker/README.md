# f8pydetect_tracker

Feel8 ONNX detector + tracker runtime service.

Status:
- Deprecated for new pipelines.
- Prefer `f8.dl.detector` / `f8.dl.humandetector` (from `f8pydl`) + `f8.cvkit.tracking` when tracking is needed.

Notes:
- This repo currently uses CPython 3.14 in some environments; `onnxruntime` wheels may not exist for that version yet.
- For NVIDIA GPU acceleration, install `onnxruntime-gpu` (supported Python) and ensure `CUDAExecutionProvider` is available.
