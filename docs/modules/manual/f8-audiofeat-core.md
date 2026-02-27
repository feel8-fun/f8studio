### Recommended Use Cases

- Compute real-time base audio descriptors from `f8.audiocap` shared memory.
- Feed rhythm/beat services with reusable onset envelope features.

### Minimal Run Example

```bash
pixi run f8pyaudiofeat_core --service-id audiofeat_core
```

### Common Pitfalls

- `audioShmName` not set or pointing to a non-existing segment.
- Window/hop configuration too small can create unstable centroid/onset output.

### Troubleshooting

- Confirm upstream `f8.audiocap` reports non-zero `writeSeq`.
- Start with defaults (`windowMs=768`, `hopMs=64`) before tuning.
