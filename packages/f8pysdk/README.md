## f8pysdk

Python runtime SDK for running an F8 service process.

### Core building blocks
- `ServiceBus` (`f8pysdk.bus`): public runtime facade over transport, routing, state, commands, and lifecycle.
- `ServiceHost` (`f8pysdk.host`): rungraph-driven runtime node materialization/registration.
- `ServiceRuntime` (`f8pysdk.runtime`): runtime facade that wires `ServiceBus` + `ServiceHost`.

Core stable public modules:
- `f8pysdk.app`: high-level service app entrypoint
- `f8pysdk.bus`: `ServiceBus`, `ServiceBusConfig`, and explicit bus component factory types
- `f8pysdk.host`: `ServiceHost` and `ServiceHostConfig`
- `f8pysdk.runtime`: `ServiceRuntime` and `ServiceRuntimeConfig`
- `f8pysdk.specs`: generated protocol models, schema helpers, and spec metadata/edit-policy helpers
- `f8pysdk.nodes`: `RuntimeNode`, `ServiceNode`, `OperatorNode`
- `f8pysdk.registry`: author-facing `Registry`, low-level `RuntimeNodeRegistry`, and registry errors
- `f8pysdk.command`: command result/output policy types
- `f8pysdk.data`: data delivery and cross-publish policy types
- `f8pysdk.monitoring`: monitor snapshot collection/config types
- `f8pysdk.state`: canonical state read/write types
- `f8pysdk.runtime_transport`: backend-neutral transport protocol
- `f8pysdk.zenoh_transport`: default Zenoh runtime transport
- `f8pysdk.testing`: in-memory harness plus emit/pull/buffer test helpers

Additional stable utility modules:
- `f8pysdk.zenoh_naming`: canonical Zenoh key-expression helpers
- `f8pysdk.f8_naming`: canonical runtime token, subject, and state-key helpers
- `f8pysdk.capabilities`: explicit runtime node capability protocols and mixins
- `f8pysdk.rungraph_validation`: rungraph validation helpers used by Studio/runtime tooling
- `f8pysdk.editor_assist_protocol`: editor-assist payload validation helpers
- `f8pysdk.time_utils`: small runtime timestamp helpers
- `f8pysdk.service_runtime_tools`: advanced tooling namespace
  import from explicit owner subpackages:
  `service_runtime_tools.inventory.*` and `service_runtime_tools.deploy.*`
  the old flat helper paths under `service_runtime_tools.*` have been removed
  within `inventory`, use `entry` for `service.yml` loading/roots, `describe` for describe diagnostics, and `discovery` for catalog-loading orchestration

Explicit internal-only modules:
- `f8pysdk.service_bus.internal.*`: repo-internal runtime/test helpers owned by the ServiceBus runtime and not part of the public SDK API
  this replaces the earlier top-level pseudo-internal boundary so ownership stays with the runtime subsystem that defines these helpers
- `f8pysdk._specs.*`: repo-internal spec-shaping helpers owned by the `specs` subsystem
  builtin describe/state-field shaping now lives under `f8pysdk._specs.builtin_fields` instead of a top-level helper module

Removed deep legacy paths:
- old `f8pysdk.service_bus.*` compatibility shims such as `bus`, `codec`, `command_runtime`, `cross_state`, `domain.state_pipeline`, `error_utils`, `lifecycle`, `metadata`, `micro`, `payload`, `routing.data_*`, `routing_data`, `rungraph_apply`, `runtime_collections`, `state_publish`, `state_router`, and `state_store` have been removed from this repo
- historical top-level implementation paths such as `f8pysdk.runtime_node`, `f8pysdk.runtime_node_registry`, `f8pysdk.service_host`, `f8pysdk.service_runtime`, `f8pysdk.msgspec_codec`, `f8pysdk.nats_transport`, `f8pysdk.command_state`, `f8pysdk.json_unwrap`, and `f8pysdk.monitor_schema` have been removed in favor of stable owner modules like `f8pysdk.nodes`, `f8pysdk.registry`, `f8pysdk.host`, `f8pysdk.runtime`, `f8pysdk.codec`, `f8pysdk.command`, and `f8pysdk.monitoring`
- top-level helper modules `f8pysdk.builtin_state_fields`, `f8pysdk.service_ready`, and `f8pysdk.nats_server_bootstrap` have been removed
  use `f8pysdk._specs.builtin_fields` and `f8pysdk.service_runtime_tools.deploy.readiness` instead
- repo code should import stable SDK modules such as `f8pysdk.bus`, `f8pysdk.specs`, `f8pysdk.command`, `f8pysdk.data`, `f8pysdk.state`, `f8pysdk.codec`, and `f8pysdk.testing` directly
- package root `f8pysdk` no longer wildcard-reexports generated types or helper functions; import from explicit owner modules instead

Internal runtime split:
- `DataRouter`: data routing, buffering, subscriptions, callback batching
- `StateStore`: state cache, access maps, KV-backed reads
- `StateRouter`: intra-state routing and cross-state watch lifecycle
- `CommandGateway`: canonical command execution path
- `service_bus.runtime`: canonical `ServiceBus` owner
- `service_bus.config`: canonical `ServiceBusConfig` owner
- `service_bus.data.*`: owner package for data runtime internals
- `service_bus.state.*`: owner package for state runtime internals
- `service_bus.state.options`: internal owner for `StatePublishOptions`
- `service_bus.state.pipeline`: state validation, persistence, and local-delivery owner
- `service_bus.workflow.metadata`: workflow-owned metadata builders
- `service_bus.internal.cache` / `service_bus.internal.logging`: shared runtime infrastructure helpers

Lifecycle contract:
- `ServiceBus` and `ServiceRuntime` are single-run objects.
- After `stop()`, create a new instance instead of calling `start()` again.

### Recommended service entrypoint
Prefer `ServiceApp` plus `Registry` for new services:
- one explicit owner object for describe/build/run/CLI
- one explicit registry owner for specs and runtime node factories
- standard CLI: `--describe`, `--service-id`, `--bus-backend`, and Zenoh options
- default backend: `zenoh`
- explicit local-test fallback: `--bus-backend mem`

Minimal example:

```py
from f8pysdk.app import ServiceApp
from f8pysdk.registry import Registry
from mysvc.specs import MY_OPERATOR_SPEC, MY_SERVICE_SPEC
from mysvc.nodes import MyOperatorNode, MyServiceNode

registry = Registry()


def register_specs(registry: Registry) -> Registry:
    registry.register_service(MY_SERVICE_SPEC, MyServiceNode)
    registry.register_operator(MY_OPERATOR_SPEC, MyOperatorNode)
    return registry


register_specs(registry)

app = ServiceApp(
    service_class="f8.myservice",
    registry=registry,
)

app.run(service_id="svc-a")

# Local in-memory tests:
# app.run(service_id="svc-a", bus_backend="mem")
```

Registry contract:
- `Registry()` creates a fresh process-local runtime registry owner by default.
- Use `ServiceApp.build_shared_registry()` or `shared_registry()` only when process-global sharing is intentional.
- `Registry.register_service(spec, node_type_or_factory)` pairs describe-time metadata with runtime node construction in one call.
- `Registry.register_operator(spec, node_type_or_factory)` does the same for operators.
- Pass an explicit registry into `ServiceRuntime(...)` or `ServiceHost(...)` when multiple components must share registration state.
- Keep service/operator modules on `def register_specs(registry: Registry) -> Registry` or `def register_operator(registry: Registry) -> Registry`; return the same registry for chaining/composition.
- Use `RuntimeNodeRegistry` and the lower-level `register_*_spec(...)` / `register_*_factory(...)` calls only at runtime internals, compatibility boundaries, or focused tests.
- Missing operator runtime factories now fail explicitly; generic `ServiceNode` remains the default container when a service class has no custom service factory.

Runtime config contract:
- `ServiceBusConfig` is the single runtime configuration core for service id/class, transport, routing, data delivery, cache limits, and monitor options.
- `ServiceRuntimeConfig` only composes `bus: ServiceBusConfig` plus `registry_modules`.
- `ServiceHost` derives its service class from `ServiceBusConfig.service_class` unless an explicit `ServiceHostConfig` override is passed.
- `ServiceAppDefaults` stores a `ServiceBusConfig` template and turns it into a concrete runtime bus config when `service_id`, `service_class`, and CLI/env overrides are known.

Compatibility note:
- repo entrypoints now use `ServiceApp`; new code should follow that single explicit app owner model.
- lower-level runtime composition should import `f8pysdk.host` and `f8pysdk.runtime`; the historical `service_host.py` / `service_runtime.py` implementation files have been removed.

Command contract:
- `ServiceBus.invoke_command(node_id, call, args, ...)` is the explicit local command API.
- `ServiceBus.invoke_command(...)` is reply-first by default and does not write hidden output state unless asked.
- Declared commands normalize scalar/list/dict args from their `F8Command.params` definition.
- Hidden command state fields remain supported as the graph command adapter.
- Use `output_policy=CommandOutputPolicy.hidden_state` when a caller explicitly wants hidden output state writeback.

Data contract:
- `ServiceBus.emit_data(node_id, port, value, ...)` is the canonical public data-output API.
- `cross_publish_policy` controls network fan-out explicitly: `routed`, `all`, or `none`.
- `data_delivery` controls local consumer shape explicitly: `callback` or `buffered`.
- Default runtime behavior is `cross_publish_policy="routed"` and `data_delivery="callback"`.
- `ServiceBus.pull_data(node_id, port, ...)` reads local buffered inputs and may trigger same-service upstream compute when needed.
- Pull-triggered local compute satisfies local consumers only; internal typed emit options keep it from turning into hidden cross-service traffic.
- Monitor snapshots now expose routing counters in `frame`, including local-only emits, routed cross emits, suppressed cross publishes, callback deliveries, and buffered pull deliveries.
- Invalid delivery modes fail fast; legacy `push` / `pull` / `both` aliases are not accepted.

Stable helper modules:
- `f8pysdk.codec`: msgpack request/reply helpers plus shared primitive coercion helpers (`encode_obj`, `decode_obj`, `decode_as`, `coerce_*`, `parse_*`)
- `f8pysdk.specs`: generated protocol models plus schema/spec helpers
- `f8pysdk.state`: canonical state result/error/context types
- `f8pysdk.command`: command output/result enums and result envelopes
- `f8pysdk.data`: data delivery and cross-publish policy types
- `f8pysdk.monitoring`: monitor collector façade
- `f8pysdk.nodes`: runtime node base classes
- `f8pysdk.registry`: author-facing registry, low-level runtime registry, and registry errors
- `f8pysdk.runtime_transport`: backend-neutral runtime transport protocol
- `f8pysdk.zenoh_transport`: default Zenoh runtime transport
- `f8pysdk.testing`: in-memory harness plus emit/pull/buffer helpers for tests

Coercion contract:
- `parse_*` means “best-effort parse”. It returns `None` when the input is missing or invalid, so callers can decide what to do next.
- `coerce_*` means “produce a usable value here”. It falls back to an explicit `default` and may also clamp/bound the result.
- Prefer `parse_*` at validation or branching points where `None` is meaningful.
- Prefer `coerce_*` at config/state consumption points where runtime code needs a concrete value.
- Avoid naming a helper `parse_*` if it always returns a fallback value; that is `coerce_*` semantics.
