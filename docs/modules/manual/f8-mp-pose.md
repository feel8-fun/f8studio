### Recommended Use Cases

- Realtime single-person pose extraction for control pipelines.
- 2D/3D skeleton feed for visualizers and mapping operators.

### Minimal Run Example

```bash
pixi run -e mediapipe mp_pose
```

### Common Pitfalls

- Source frames not mapped into expected SHM channel.
- Confidence thresholds too strict for low-light scenes.

### Troubleshooting

- Verify `shmName` and input frame producer are active.
- Lower detection/tracking thresholds and profile CPU load.
