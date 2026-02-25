### Recommended Use Cases

- Capture microphone or loopback/system audio for downstream analysis.
- Publish waveform features to visualization or control operators.

### Minimal Run Example

```bash
build/bin/f8audiocap_service.exe --service-id audiocap --mode capture --backend wasapi
```

### Common Pitfalls

- Wrong backend/device selection causes silent capture.
- Loopback capture may require platform-specific permissions.

### Troubleshooting

- Use `--list-devices` first to identify valid input device IDs.
- Verify service state fields for sample rate and channel configuration.
