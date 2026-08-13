# f8pydl

Feel8 split DL services (classification + detection + human detection), powered by ONNX Runtime.

Service classes:
- `f8.dl.classifier`
- `f8.dl.detector`
- `f8.dl.humandetector`
- `f8.dl.detsorter`
- `f8.dl.optflow`
- `f8.dl.tcnwave`

Output schemas:
- `f8visionClassifications/1` on `classifications`
- `f8visionDetections/1` on `detections`
- `f8visionDetections/1` in/out on `detections` (`f8.dl.detsorter`, reordered by score-map metric)
- `flow2_f16` latest-frame output on the `flow` data port (`f8.dl.optflow`)
- `number` on `predictedChange` (`f8.dl.tcnwave`)

Detection sorter notes (`f8.dl.detsorter`):
- Reads score maps from the typed `score` data input port.
- Supports `scalar1_f32` directly and `flow2_f16` by converting flow vectors to magnitude.
- If detection payload `width`/`height` differs from score-map `width`/`height`, bbox coordinates are rescaled to score-map space before scoring.
- Sort behavior is controlled by `sortDirection` (`desc` default, or `asc`) and `scoreAggregation` (`mean` default, or `max` / `sum` / `median`).
- `temperature` adds Gumbel/softmax-style sampling without replacement after scores are normalized. `0` (default) preserves deterministic sorting; larger values produce increasingly random orders while retaining the configured score preference.
- Keeps original detection `score` values and only reorders `detections`.

Weight YAML notes (`f8onnxModel/1`):
- `model.skeletonProtocol` controls detection payload field `skeletonProtocol`.
- `model.task: optflow_neuflowv2` enables NeuFlowV2 optical flow service.
- `model.task: tcn_wave` enables temporal wave inference service.
- `optflow.flowFormat` currently must be `flow2_f16`.
- `optflow.inputOrder` supports `prev_now` / `now_prev`.
- For `f8.dl.tcnwave`, `outputScale` / `outputBias` are node state fields.
- `useVrFocusCrop` is a node state field (runtime switch). When enabled, it crops top 20% and left/right 10% before preprocessing.
- Missing ONNX files can be auto-downloaded when `autoDownloadWeights=true`.
- Remote fields:
  - Preferred: `model.onnxUrl`, `model.onnxSHA256`
  - Legacy aliases: `model.url`, `model.sha256`, `onnxUrl`, `onnxSHA256`, `url`, `sha256`
- Recommended values:
  - pose models: `coco17`
  - non-skeleton models: `none`
- Unknown protocol strings are allowed and passed through; visualization may fall back to points-only.
