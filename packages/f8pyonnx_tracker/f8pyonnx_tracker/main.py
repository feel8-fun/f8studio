from __future__ import annotations

from f8pysdk.runtime_node_registry import RuntimeNodeRegistry
from f8pysdk.service_cli import ServiceCliTemplate

from .constants import SERVICE_CLASS
from .node_registry import register_onnxtracker_specs


class OnnxTrackerService(ServiceCliTemplate):
    @property
    def service_class(self) -> str:
        return SERVICE_CLASS

    def register_specs(self, registry: RuntimeNodeRegistry) -> None:
        register_onnxtracker_specs(registry)


def main(argv: list[str] | None = None) -> int:
    return OnnxTrackerService().cli(argv, program_name=SERVICE_CLASS)


if __name__ == "__main__":
    raise SystemExit(main())

