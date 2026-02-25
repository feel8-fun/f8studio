### Recommended Use Cases

- Primary object detection service for tracking and control workflows.
- Foundation block for scenario pipelines using bounding boxes.

### Minimal Run Example

```bash
pixi run -e onnx dl_detector
```

### Common Pitfalls

- Model confidence thresholds set too high can hide all detections.
- Wrong skeleton protocol assumptions break downstream visualizers.

### Troubleshooting

- Adjust confidence thresholds incrementally.
- Confirm `model.skeletonProtocol` in weight YAML matches consumer expectations.
