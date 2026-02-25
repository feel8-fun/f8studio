### Recommended Use Cases

- Category classification over selected regions or full frames.
- Semantic labeling step after detection/tracking.

### Minimal Run Example

```bash
pixi run -e onnx dl_classifier
```

### Common Pitfalls

- Wrong model YAML path or schema mismatch blocks startup.
- Preprocessing mismatch (size/normalization) degrades outputs.

### Troubleshooting

- Verify `f8onnxModel/1` YAML fields and model file paths.
- Start with known-good weights from `services/f8/dl/weights`.
