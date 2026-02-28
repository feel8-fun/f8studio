### Recommended Use Cases

- Build operator pipelines for signal processing and device control.
- Bridge service outputs into composable runtime logic.

### Minimal Run Example

```bash
pixi run -e default engine
```

### Operator Composition Notes

- Keep time-base generation (`tick`, `phase`, `program_wave`) separated from mapping stages (`range_map`, `smooth_filter`, `rate_limiter`).
- Use dedicated adapter operators (`lovense_program_adapter`, `buttplug_bridge`) to isolate protocol translation from signal logic.
- `f8.expr` is an operator inside `f8.pyengine`; if you need a standalone service-level expression runtime, use `f8.pyexpr`.
- For reusable chains, keep clear input/output contracts and avoid hidden side effects.

### Common Pitfalls

- Mixed push/pull data delivery can cause unintended ordering assumptions.
- Overly dense operator graphs make runtime diagnosis difficult.

### Troubleshooting

- Start with a minimal chain and add one operator at a time.
- Inspect service and operator state ports to locate stale or invalid values.
