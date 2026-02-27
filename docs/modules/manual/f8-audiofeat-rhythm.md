### Recommended Use Cases

- Derive tempo and pulse clarity from `coreFeatures` output.
- Build rungraph controls synced with music beat intensity.

### Minimal Run Example

```bash
pixi run f8pyaudiofeat_rhythm --service-id audiofeat_rhythm
```

### Common Pitfalls

- No incoming `coreFeatures` edge means rhythm node stays idle.
- Too-short `tempoWindowSec` may produce unstable BPM estimates.

### Troubleshooting

- Verify rungraph edge: `f8.audiofeat.core/coreFeatures -> f8.audiofeat.rhythm/coreFeatures`.
- Increase `tempoWindowSec` when BPM jumps too frequently.
