# f8pydl

Feel8 split DL services (classification + detection + human detection), powered by ONNX Runtime.

Service classes:
- `f8.dl.classifier`
- `f8.dl.detector`
- `f8.dl.humandetector`
- `f8.dl.optflow`

Output schemas:
- `f8visionClassifications/1` on `classifications`
- `f8visionDetections/1` on `detections`
- `flow2_f16` SHM on state `flowShmName` (`f8.dl.optflow`)

Weight YAML notes (`f8onnxModel/1`):
- `model.skeletonProtocol` controls detection payload field `skeletonProtocol`.
- `model.task: optflow_neuflowv2` enables NeuFlowV2 optical flow service.
- `optflow.flowFormat` currently must be `flow2_f16`.
- `optflow.inputOrder` supports `prev_now` / `now_prev`.
- Missing ONNX files can be auto-downloaded when `autoDownloadWeights=true`.
- Remote fields:
  - Preferred: `model.onnxUrl`, `model.onnxSHA256`
  - Legacy aliases: `model.url`, `model.sha256`, `onnxUrl`, `onnxSHA256`, `url`, `sha256`
- Recommended values:
  - pose models: `coco17`
  - non-skeleton models: `none`
- Unknown protocol strings are allowed and passed through; visualization may fall back to points-only.
