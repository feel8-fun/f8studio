### Recommended Use Cases

- Feed local media into shared memory for detection/pose services.
- Validate end-to-end pipelines with deterministic playback.

### Minimal Run Example

```bash
pixi run -e default python scripts/run_cpp_service.py --service-dir services/f8/implayer --exe f8implayer_service --env-var F8IMPLAYER_EXE
```

### Common Pitfalls

- Missing codec/runtime libraries on target host.
- Wrong SHM name wiring disconnects downstream consumers.

### Troubleshooting

- Verify media playback locally before integrating downstream nodes.
- Confirm `shmName` is identical across producer and consumers.
