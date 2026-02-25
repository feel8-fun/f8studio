### Recommended Use Cases

- Stabilize shaky input before detection and pose inference.
- Improve downstream consistency in handheld or moving-camera feeds.

### Minimal Run Example

```bash
pixi run -e default python scripts/run_cpp_service.py --service-dir services/f8/cvkit/video_stab --exe f8cvkit_video_stab_service --env-var F8CVKIT_VIDEO_STAB_EXE
```

### Common Pitfalls

- Over-aggressive settings can introduce lag artifacts.
- Scene cuts can break temporal assumptions.

### Troubleshooting

- Tune smoothing windows conservatively first.
- Reset state around hard scene transitions.
