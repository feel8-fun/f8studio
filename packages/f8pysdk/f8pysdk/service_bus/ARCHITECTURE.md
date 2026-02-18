# Service Bus Architecture

## Layered Modules

- `api/`: public entrypoint and configuration.
  - `api/bus.py`: `ServiceBus` facade and orchestration.
  - `api/config.py`: runtime configuration and delivery mode types.
  - `api/types.py`: stable state read/write types.
- `domain/`: state-write policy and pipeline.
  - `domain/state_pipeline.py`: normalize, validate, dedupe, persist, local deliver, fanout.
- `routing/`: data-route and buffer mechanics.
  - `routing/data_flow.py`: input buffers, pull/push behavior, routed subscriptions.
- `workflow/`: rungraph/lifecycle/cross-state workflows.
  - `workflow/rungraph.py`: apply/validate/rebuild/init sync.
  - `workflow/cross_state.py`: remote watch sync and cross-service state propagation.
  - `workflow/lifecycle.py`: start/stop/ready/active workflow.
- `adapters/`: infrastructure-specific integration.
  - `adapters/micro.py`: NATS micro endpoints.

## Compatibility Layer

Legacy top-level modules are preserved as compatibility facades and re-export the new layered implementation:

- `bus.py`, `state_publish.py`, `routing_data.py`, `rungraph_apply.py`, `cross_state.py`, `lifecycle.py`, `micro.py`

This keeps imports stable while allowing internal refactoring by layer.

## Dependency Rule

- `api` can depend on `workflow/domain/routing/adapters`.
- `workflow` can depend on `domain/routing/adapters`.
- `domain` and `routing` must not depend on `workflow`.
- `adapters` should not contain domain policy.
