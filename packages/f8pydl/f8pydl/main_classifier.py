from __future__ import annotations

from f8pysdk.runtime_node_registry import RuntimeNodeRegistry
from f8pysdk.service_cli import ServiceCliTemplate

from .constants import CLASSIFIER_SERVICE_CLASS
from .node_registry import register_specs


class DlClassifierService(ServiceCliTemplate):
    @property
    def service_class(self) -> str:
        return CLASSIFIER_SERVICE_CLASS

    def register_specs(self, registry: RuntimeNodeRegistry) -> None:
        register_specs(registry)


def main(argv: list[str] | None = None) -> int:
    return DlClassifierService().cli(argv, program_name=CLASSIFIER_SERVICE_CLASS)


if __name__ == "__main__":
    raise SystemExit(main())
