## f8pysdk

Python runtime SDK for running an F8 service process.

### Core building blocks
- `ServiceBus` (`f8pysdk/service_bus.py`): NATS + JetStream KV transport, routing tables, state cache, rungraph watch.
- `ServiceHost` (`f8pysdk/service_host.py`): rungraph-driven runtime node materialization/registration.
- `ServiceRuntime` (`f8pysdk/service_runtime.py`): runtime facade that wires `ServiceBus` + `ServiceHost`.

### Recommended “fill-in-the-blanks” entrypoint
Use `ServiceCliTemplate` (`f8pysdk/service_cli.py`) to keep each service process consistent:
- standard CLI: `--describe`, `--service-id`, `--nats-url` (with `F8_SERVICE_ID`, `F8_NATS_URL` env fallbacks)
- fixed lifecycle hooks:
  - `register_specs(registry)` (required)
  - `setup(app)` (optional)
  - `teardown(app)` (optional)

Minimal example:

```py
from f8pysdk.service_cli import ServiceCliTemplate
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

class MyService(ServiceCliTemplate):
    @property
    def service_class(self) -> str:
        return "f8.myservice"

    def register_specs(self, registry: RuntimeNodeRegistry) -> None:
        # registry.register_service(...)
        # registry.register(...)
        pass
```

### Headless Runner

Run a saved Studio session JSON without launching UI:

```bash
python -m f8pysdk.headless_runner --session path/to/session.json
```

Default behavior:
- attempts local NATS bootstrap (auto-start/download) when needed
- auto-starts discovered service processes
- deploys per-service rungraphs and waits until termination signal

Useful flags:
- `--no-bootstrap`: disable NATS auto bootstrap
- `--no-auto-start`: deploy only to already-running services
