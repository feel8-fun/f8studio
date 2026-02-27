# DL Weights Directory

Place model yaml + onnx pairs here for:
- `f8.dl.classifier`
- `f8.dl.detector`
- `f8.dl.humandetector`
- `f8.dl.optflow`
- `f8.dl.tcnwave`

Supported model yaml schema:
- `f8onnxModel/1`
- `task: yolo_cls` is supported for classification models (see `example_yolo_cls.yaml`).
- `task: optflow_neuflowv2` is supported for optical flow models (see `neuflow_mixed.yaml`).
- `task: tcn_wave` is supported for temporal wave models.
- Optional remote fetch fields:
  - Preferred: `model.onnxUrl`, `model.onnxSHA256`
  - Legacy aliases (still supported): `model.url`, `model.sha256`, `onnxUrl`, `onnxSHA256`, `url`, `sha256`
  - Services can auto-download missing files when `autoDownloadWeights=true`.

Runtime node state (for `f8.dl.tcnwave`):
- `outputScale` controls output denormalization scale (default `10.0`).
- `outputBias` controls output denormalization bias (default `0.0`).
- `useVrFocusCrop` controls focus cropping at runtime (top 20% + left/right 10%).
