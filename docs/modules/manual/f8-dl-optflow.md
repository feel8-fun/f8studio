### Recommended Use Cases

- GPU optical-flow inference with NeuFlowV2 ONNX models.
- Dense motion SHM generation for downstream script/operator consumption.

### Minimal Run Example

```bash
pixi run f8pydl_optflow
```

### Common Pitfalls

- `inputShmName` mismatch causes no output flow frames.
- Missing `.onnx` model file or wrong `task` in weights YAML prevents runtime init.

### Troubleshooting

- Verify `flowShmName` is populated and `flowShmFormat` is `flow2_f16`.
- Start with `computeEveryNFrames=2`, then tune throughput/latency.
