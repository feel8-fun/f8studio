### Script Hooks

`f8.pyscript` provides one in-process script runtime per service instance.

Supported hooks:
- `onStart(ctx)` / `onStop(ctx)`
- `onPause(ctx, meta)` / `onResume(ctx, meta)`
- `onState(ctx, field, value, tsMs)`
- `onData(ctx, port, value, tsMs)`
- `onTick(ctx, tick)`
- `onCommand(ctx, name, args, meta)`

`ctx['exec_local'](...)` is gated by command `grant_local_exec` and can be revoked via `revoke_local_exec`.

State helpers:
- `await ctx['get_state'](field)` for freshest reads.
- `ctx['get_state_cached'](field, default=None)` for sync cached snapshots in hot paths.
