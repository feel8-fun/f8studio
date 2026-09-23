## When to Use

- Use `Switch Mixer` when your graph needs to select among multiple signal channels or tracks while keeping the transition smooth and controlled.
- It is a good fit for state-driven routing such as primary vs fallback motion, manual override vs automatic control, or switching among several behavior profiles.
- Best for graphs where direct hard switching would feel too abrupt and you want the choice of a short crossfade instead.

## Common Wiring Patterns

- **Named Channel Routing**: Add custom data input ports such as `main`, `fallback`, `manual`, or `track_3`, then drive `currentChannel` from graph state or UI.
- **Primary / Fallback Routing**: Pair `Silence Detector.isSilent` with graph logic that updates `currentChannel` to a fallback port when the main source becomes inactive.
- **Soft Transition Monitoring**: Keep a `Wave Viz` after the mixer and inspect `alpha` to verify that transitions are smooth instead of hard cuts.

## Pitfalls / Gotchas

- **Input Range Mismatch**: If your channels are not normalized to similar ranges, the transition can still feel jumpy even with crossfade enabled.
- **Missing Control Logic**: `Switch Mixer` does not decide when to switch; pair it with explicit state logic such as `Silence Detector`, `State Expr`, or UI-driven state.
- **Hold Semantics**: When a selected channel stops receiving valid samples, the mixer holds that channel's last valid value. That is useful for continuity, but it also means stale upstream data can remain audible if your graph never switches away.
- **Too Much Fade**: Very large `fadeMs` values can make the graph feel sluggish or indecisive when the control condition changes quickly.
