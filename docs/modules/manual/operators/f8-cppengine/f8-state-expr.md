## When to Use

- Use `State Expr` when you need to derive a computed value or "virtual property" from other editable state fields on the same node, rather than from incoming data ports.
- It is excellent for creating formulas, derived parameters, and lightweight control math that should stay visible and tunable directly on the node's property panel.
- Use it to enforce relationships between parameters (e.g., `max_speed = base_speed * multiplier`).

## Common Wiring Patterns

- **Derived Parameters**: Add multiple numeric state fields (properties) as your input variables. The operator automatically exposes these as symbols in your expression. Publish the result (`out`) to other nodes or use it for Studio inspection.
- **Live Tuning**: Keep the expression focused on a small set of clearly named fields so the system remains easy to calibrate during a live session.
- **Inspector Feedback**: Use it to create "Read-only" calculated fields that summarize the current state of a complex operator group for easier monitoring.

## Pitfalls / Gotchas

- **Type Restrictions**: Only writable numeric state fields (float, int) are automatically extracted as symbols. Non-numeric fields or read-only properties will not be available in the expression.
- **Error Visibility**: If an expression fails (e.g., division by zero), the operator reports the error through the monitor alert channel and clears the output. If your downstream graph seems "stuck," check Studio notifications or the service monitor report.
- **Circular Dependencies**: Be careful not to create logical loops where an expression depends on a value that is eventually affected by its own output, as this can lead to unstable behavior.
