### Recommended Use Cases

- Follow detector-provided bounding boxes through time.
- Build persistent target state for downstream control chains.

### Minimal Run Example

```bash
pixi run -e default python scripts/run_cpp_service.py --service-dir services/f8/cvkit/tracking --exe f8cvkit_tracking_service --env-var F8CVKIT_TRACKING_EXE
```

### Common Pitfalls

- Invalid `initBox` schema prevents tracker startup.
- Frequent detector resets can fight tracker state.

### Troubleshooting

- Validate detector output shape before routing to `initBox`.
- Use `stopTracking` command to recover from stale state.
