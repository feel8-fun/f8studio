## When to Use

- Use this declaration when planning stable character selection by exporter profile, role, and role index.
- The native runtime is pending. Use PyEngine's `Skeleton Selector` to process live streams.

## Common Wiring Patterns

- Connect decoded skeleton lists to `skeletons` and the selected `skeleton` to a bone selector.
- Set `profileId`, `role`, and zero-based `roleIndex` for the desired character.
- For legacy streams, declare `fallbackModelName` and enable `allowLegacyFallback` explicitly.

## Pitfalls / Gotchas

- The current C++ implementation reports `CPP_OPERATOR_UNIMPLEMENTED` and returns null data outputs.
- Model names do not establish stable semantic roles in new exporter protocols.
- Keep selection status on the data channel and treat missing characters as invalid input.
