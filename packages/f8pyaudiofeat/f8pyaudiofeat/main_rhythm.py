from __future__ import annotations

from f8pysdk.runtime_node_registry import RuntimeNodeRegistry
from f8pysdk.service_cli import ServiceCliTemplate
from f8pysdk.service_runtime import ServiceRuntimeConfig

from .constants import RHYTHM_SERVICE_CLASS
from .node_registry import register_specs


class AudioFeatureRhythmService(ServiceCliTemplate):
    @property
    def service_class(self) -> str:
        return RHYTHM_SERVICE_CLASS

    def register_specs(self, registry: RuntimeNodeRegistry) -> None:
        register_specs(registry)

    def build_runtime_config(self, *, service_id: str, nats_url: str) -> ServiceRuntimeConfig:
        return ServiceRuntimeConfig.from_values(
            service_id=service_id,
            service_class=self.service_class,
            nats_url=nats_url,
            data_delivery="both",
        )


def main(argv: list[str] | None = None) -> int:
    return AudioFeatureRhythmService().cli(argv, program_name=RHYTHM_SERVICE_CLASS)


if __name__ == "__main__":
    raise SystemExit(main())
