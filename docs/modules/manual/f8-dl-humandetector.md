### Recommended Use Cases

- Specialized human detection pipeline for person-centric interactions.
- Front-end filtering before pose or identity stages.

### Minimal Run Example

```bash
pixi run -e onnx dl_humandetector
```

### Common Pitfalls

- Running on unsupported GPU stack may fallback unexpectedly.
- Inconsistent input aspect ratio can reduce accuracy.

### Troubleshooting

- Validate ONNX Runtime GPU availability in the selected environment.
- Normalize input resolution/aspect ratio before inference.
