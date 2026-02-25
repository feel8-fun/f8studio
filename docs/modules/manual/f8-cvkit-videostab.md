### Recommended Use Cases

- Stabilize shaky input before detection and pose inference.
- Improve downstream consistency in handheld or moving-camera feeds.

### Minimal Run Example

```bash
services/f8/cvkit/linux/f8cvkit_video_stab_service
```

### Common Pitfalls

- Over-aggressive settings can introduce lag artifacts.
- Scene cuts can break temporal assumptions.

### Troubleshooting

- Tune smoothing windows conservatively first.
- Reset state around hard scene transitions.
