### Recommended Use Cases

- Motion field extraction for gesture/velocity-aware interactions.
- Preprocessing stage before higher-level temporal analytics.

### Minimal Run Example

```bash
services/f8/cvkit/linux/f8cvkit_dense_optflow_service
```

### Common Pitfalls

- Large frame sizes increase latency.
- Unstable input frame rates reduce flow quality.

### Troubleshooting

- Start with smaller resolutions.
- Confirm upstream frame source timestamps are monotonic.
