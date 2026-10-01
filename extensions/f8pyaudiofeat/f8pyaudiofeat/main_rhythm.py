from __future__ import annotations

from f8pysdk.app import ServiceApp, ServiceAppDefaults
from f8pysdk.bus import ServiceBusConfig
from f8pysdk.registry import Registry

from .constants import RHYTHM_SERVICE_CLASS
from .node_registry import register_specs


def build_app() -> ServiceApp:
    registry = Registry()
    register_specs(registry)
    return ServiceApp(
        service_class=RHYTHM_SERVICE_CLASS,
        registry=registry,
        defaults=ServiceAppDefaults(bus=ServiceBusConfig(data_delivery="callback")),
    )


def main(argv: list[str] | None = None) -> int:
    return build_app().cli(argv, program_name=RHYTHM_SERVICE_CLASS)


if __name__ == "__main__":
    raise SystemExit(main())
