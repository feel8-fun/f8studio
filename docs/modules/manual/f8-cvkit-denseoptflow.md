### Recommended Use Cases

- Motion field extraction for gesture/velocity-aware interactions.
- Preprocessing stage before higher-level temporal analytics.

### Minimal Run Example

```bash
pixi run -e default python scripts/run_cpp_service.py --service-dir services/f8/cvkit/dense_optflow --exe f8cvkit_dense_optflow_service --env-var F8CVKIT_DENSE_OPTFLOW_EXE
```

### Common Pitfalls

- Large frame sizes increase latency.
- Unstable input frame rates reduce flow quality.

### Troubleshooting

- Start with smaller resolutions.
- Confirm upstream frame source timestamps are monotonic.
