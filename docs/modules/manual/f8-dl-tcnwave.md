### Recommended Use Cases

- Sequence-to-wave regression from motion feature windows.
- Producing scalar prediction streams for downstream control mapping.

### Minimal Run Example

```bash
pixi run -e onnx f8pydl_tcnwave
```

### Common Pitfalls

- Model input window length mismatch causes runtime validation errors.
- Wrong weights YAML path prevents ONNX session initialization.

### Troubleshooting

- Verify `weights` points to a valid config under `services/f8/dl/weights`.
- Check `predictedChange` output updates while input windows are arriving.
