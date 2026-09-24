## When to Use

- Use `Tick` when your graph needs a simple, periodic execution (exec) trigger to drive deterministic update loops.
- It is the standard root node for periodic `f8.cppengine` logic chains, independent of the browser frame rate.
- Ideal for polling sensors, updating state machines, or sending periodic commands to hardware.

## Common Wiring Patterns

- **Logic Heartbeat**: Connect the `tick` output to a `Sequence` operator to drive multiple branches (Read -> Process -> Write) in a predictable order every cycle.
- **Hardware Sync**: Align the `tickMs` property (e.g., 20ms for 50Hz) with the expected interval of your downstream device nodes (like `f8-tcode.intervalMs`) to minimize jitter.
- **Performance Branching**: Use different `Tick` nodes with different frequencies (one "fast" for motion, one "slow" for status checks) to optimize CPU usage.

## Pitfalls / Gotchas

- **Scheduling Overload**: Setting a very small `tickMs` (e.g., 1ms) can overwhelm the engine thread if the graph is complex, leading to unstable timing and high CPU usage. Aim for the minimum frequency required for smooth motion.
- **Dangling Logic**: If downstream nodes are not "exec-driven" (they don't have an input port for execution triggers), adding more `Tick` roots will not affect their update frequency.
- **Deterministic Drift**: While the tick attempts to be consistent, actual execution time depends on the system load. Monitor the `dt` (delta time) output if your logic requires millisecond-precise timing across different hardware.
