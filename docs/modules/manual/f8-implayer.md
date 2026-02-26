### Recommended Use Cases

- Feed local media into shared memory for detection/pose services.
- Validate end-to-end pipelines with deterministic playback.

### Minimal Run Example

```bash
services/f8/implayer/linux/f8implayer_service
```

### Common Pitfalls

- Missing codec/runtime libraries on target host.
- Wrong SHM name wiring disconnects downstream consumers.

### Troubleshooting

- Verify media playback locally before integrating downstream nodes.
- Confirm `shmName` is identical across producer and consumers.
