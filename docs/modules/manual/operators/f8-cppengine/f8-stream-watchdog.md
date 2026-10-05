## When to Use

- Use this declaration when designing freshness checks for timestamped streaming data.
- The native runtime is pending. Use PyEngine's `Stream Watchdog` for executable freshness gates.

## Common Wiring Patterns

- Connect timestamped input to `value` and a periodic trigger to `check`.
- Configure `timeoutMs` for the expected stream cadence.
- Use the intended `valid` exec output to gate a downstream output stage.

## Pitfalls / Gotchas

- The current C++ implementation reports `CPP_OPERATOR_UNIMPLEMENTED`, returns null data, and emits no exec outputs.
- This declaration currently provides no operational freshness protection.
- Sample age and validity belong on data or monitor channels; arming remains a separate configuration choice.
