# f8pydl

Feel8 split DL services (classification + detection + human detection), powered by ONNX Runtime.

Service classes:
- `f8.dl.classifier`
- `f8.dl.detector`
- `f8.dl.humandetector`

Output schemas:
- `f8visionClassifications/1` on `classifications`
- `f8visionDetections/1` on `detections`

Weight YAML notes (`f8onnxModel/1`):
- `model.skeletonProtocol` controls detection payload field `skeletonProtocol`.
- Recommended values:
  - pose models: `coco17`
  - non-skeleton models: `none`
- Unknown protocol strings are allowed and passed through; visualization may fall back to points-only.
