from __future__ import annotations

from f8pysdk.app import ServiceApp
from f8pysdk.registry import Registry

from .constants import CLASSIFIER_SERVICE_CLASS
from .node_registry import register_specs


def build_app() -> ServiceApp:
    registry = Registry()
    register_specs(registry)
    return ServiceApp(service_class=CLASSIFIER_SERVICE_CLASS, registry=registry)


def main(argv: list[str] | None = None) -> int:
    return build_app().cli(argv, program_name=CLASSIFIER_SERVICE_CLASS)


if __name__ == "__main__":
    raise SystemExit(main())
