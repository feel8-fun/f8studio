from __future__ import annotations

from typing import Any

from f8pysdk import (
    F8DataPortSpec,
    F8RuntimeNode,
    F8ServiceSchemaVersion,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    integer_schema,
    number_schema,
    string_schema,
)
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from .constants import CORE_SERVICE_CLASS, RHYTHM_SERVICE_CLASS
from .core_service_node import AudioCoreFeatureServiceNode
from .rhythm_service_node import AudioRhythmFeatureServiceNode


def _core_state_fields() -> list[F8StateSpec]:
    return [
        F8StateSpec(
            name="audioShmName",
            label="Audio SHM",
            description="Audio SHM mapping name (e.g. shm.audiocap.audio).",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="channelMode",
            label="Channel Mode",
            description="Channel selection for analysis.",
            valueSchema=string_schema(default="mono_mix", enum=["mono_mix", "left", "right"]),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="windowMs",
            label="Window (ms)",
            description="Feature analysis window size in milliseconds.",
            valueSchema=integer_schema(default=768, minimum=64, maximum=8000),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="hopMs",
            label="Hop (ms)",
            description="Feature analysis hop size in milliseconds.",
            valueSchema=integer_schema(default=64, minimum=8, maximum=2000),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="emitEveryHops",
            label="Emit Every Hops",
            description="Emit one coreFeatures payload every N analysis hops.",
            valueSchema=integer_schema(default=1, minimum=1, maximum=1000),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="lastError",
            label="Last Error",
            description="Last runtime error string.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.ro,
            showOnNode=False,
        ),
    ]


def _rhythm_state_fields() -> list[F8StateSpec]:
    return [
        F8StateSpec(
            name="tempoWindowSec",
            label="Tempo Window (s)",
            description="Window length in seconds for tempo estimation.",
            valueSchema=number_schema(default=8.0, minimum=1.0, maximum=60.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="pulseWindowSec",
            label="Pulse Window (s)",
            description="Window length in seconds for pulse clarity.",
            valueSchema=number_schema(default=6.0, minimum=1.0, maximum=60.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="emitEvery",
            label="Emit Every",
            description="Emit one rhythmFeatures payload every N coreFeatures inputs.",
            valueSchema=integer_schema(default=1, minimum=1, maximum=1000),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="lastError",
            label="Last Error",
            description="Last runtime error string.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.ro,
            showOnNode=False,
        ),
    ]


def _register_core(reg: RuntimeNodeRegistry) -> None:
    reg.register_service_spec(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=CORE_SERVICE_CLASS,
            version="0.0.1",
            label="Audio Feature Core",
            description="Audio SHM core feature extraction service (rms, onset, centroid).",
            tags=["audio", "feature", "rms", "onset", "centroid"],
            rendererClass="default_svc",
            stateFields=_core_state_fields(),
            dataOutPorts=[
                F8DataPortSpec(
                    name="coreFeatures",
                    description="Core feature payload with onset envelope history.",
                    valueSchema=any_schema(),
                )
            ],
            editableStateFields=False,
            editableDataInPorts=False,
            editableDataOutPorts=False,
            editableCommands=False,
        ),
        overwrite=True,
    )

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return AudioCoreFeatureServiceNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register_service(CORE_SERVICE_CLASS, _factory, overwrite=True)


def _register_rhythm(reg: RuntimeNodeRegistry) -> None:
    reg.register_service_spec(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=RHYTHM_SERVICE_CLASS,
            version="0.0.1",
            label="Audio Feature Rhythm",
            description="Rhythm analysis service consuming core features (tempo + pulse clarity).",
            tags=["audio", "feature", "tempo", "beat", "pulse"],
            rendererClass="default_svc",
            stateFields=_rhythm_state_fields(),
            dataInPorts=[
                F8DataPortSpec(
                    name="coreFeatures",
                    description="Input core feature payload from f8.audiofeat.core.",
                    valueSchema=any_schema(),
                )
            ],
            dataOutPorts=[
                F8DataPortSpec(
                    name="rhythmFeatures",
                    description="Rhythm feature payload.",
                    valueSchema=any_schema(),
                )
            ],
            editableStateFields=False,
            editableDataInPorts=False,
            editableDataOutPorts=False,
            editableCommands=False,
        ),
        overwrite=True,
    )

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return AudioRhythmFeatureServiceNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register_service(RHYTHM_SERVICE_CLASS, _factory, overwrite=True)


def register_specs(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()
    _register_core(reg)
    _register_rhythm(reg)
    return reg
