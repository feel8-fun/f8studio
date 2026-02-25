### Recommended Use Cases

- Follow detector-provided bounding boxes through time.
- Build persistent target state for downstream control chains.

### Minimal Run Example

```bash
services/f8/cvkit/linux/f8cvkit_tracking_service
```

### Common Pitfalls

- Invalid `initBox` schema prevents tracker startup.
- Frequent detector resets can fight tracker state.

### Troubleshooting

- Validate detector output shape before routing to `initBox`.
- Use `stopTracking` command to recover from stale state.
