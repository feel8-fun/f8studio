### Recommended Use Cases

- Desktop capture input for live detection and tracking workflows.
- Fast prototyping without camera hardware dependencies.

### Minimal Run Example

```bash
pixi run -e default python scripts/run_cpp_service.py --service-dir services/f8/screencap --exe f8screencap_service --env-var F8SCREENCAP_EXE
```

### Common Pitfalls

- Linux Wayland sessions are not supported by current backend.
- Capture permissions can block frame acquisition.

### Troubleshooting

- On Linux, run under X11 and validate `DISPLAY`.
- Test display enumeration before starting full scenario graph.
