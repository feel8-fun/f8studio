from __future__ import annotations

from f8pysdk.specs import (
    F8DataPortSpec,
    F8ServiceSchemaVersion,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
    array_schema,
    audio_chunk_port,
    complex_object_schema,
    integer_schema,
    number_schema,
    string_schema,
)
from f8pysdk.registry import Registry, RuntimeNodeRegistry, create_runtime_node_registry, shared_runtime_node_registry

from .constants import CORE_SERVICE_CLASS, RHYTHM_SERVICE_CLASS
from .core_service_node import AudioCoreFeatureServiceNode
from .rhythm_service_node import AudioRhythmFeatureServiceNode


def _core_features_schema():
    return complex_object_schema(
        properties={
            "schemaVersion": string_schema(),
            "tsMs": integer_schema(),
            "seq": integer_schema(),
            "sampleRate": integer_schema(),
            "hopLength": integer_schema(),
            "windowLength": integer_schema(),
            "rms": number_schema(),
            "spectralCentroidHz": number_schema(),
            "onsetStrength": number_schema(),
            "onsetEnvelope": array_schema(items=number_schema()),
        }
    )


def _rhythm_features_schema():
    return complex_object_schema(
        properties={
            "schemaVersion": string_schema(),
            "tsMs": integer_schema(),
            "seq": integer_schema(),
            "tempoBpm": number_schema(),
            "beatPeriodMs": number_schema(),
            "pulseClarity": number_schema(),
            "onsetStrengthMean": number_schema(),
            "onsetStrengthStd": number_schema(),
        }
    )


def _core_state_fields() -> list[F8StateSpec]:
    return [
        F8StateSpec(
            name="channelMode",
            label="Channel Mode",
            description="Channel selection for analysis.",
            valueSchema=string_schema(default="mono_mix", enum=["mono_mix", "left", "right"]),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="windowMs",
            label="Window (ms)",
            description="Feature analysis window size in milliseconds.",
            valueSchema=integer_schema(default=768, minimum=64, maximum=8000),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="hopMs",
            label="Hop (ms)",
            description="Feature analysis hop size in milliseconds.",
            valueSchema=integer_schema(default=64, minimum=8, maximum=2000),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="emitEveryHops",
            label="Emit Every Hops",
            description="Emit one coreFeatures payload every N analysis hops.",
            valueSchema=integer_schema(default=1, minimum=1, maximum=1000),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
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
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="pulseWindowSec",
            label="Pulse Window (s)",
            description="Window length in seconds for pulse clarity.",
            valueSchema=number_schema(default=6.0, minimum=1.0, maximum=60.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="emitEvery",
            label="Emit Every",
            description="Emit one rhythmFeatures payload every N coreFeatures inputs.",
            valueSchema=integer_schema(default=1, minimum=1, maximum=1000),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
    ]


def _register_core(registry: Registry) -> None:
    registry.register_service(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=CORE_SERVICE_CLASS,
            paletteCategory="svc",
            version="0.0.1",
            label="Audio Feature Core",
            description="Zenoh latest-audio core feature extraction service (rms, onset, centroid).",
            tags=["audio", "feature", "rms", "onset", "centroid"],
            rendererClass="default_svc",
            stateFields=_core_state_fields(),
            dataInPorts=[
                audio_chunk_port(
                    name="audio",
                    description="Input audio chunk stream from f8.audiocap.",
                    definition_protected=True,
                )
            ],
            dataOutPorts=[
                F8DataPortSpec(
                    name="coreFeatures",
                    description="Core feature payload with onset envelope history.",
                    valueSchema=_core_features_schema(),
                )
            ],
        ),
        AudioCoreFeatureServiceNode,
        overwrite=True,
    )


def _register_rhythm(registry: Registry) -> None:
    registry.register_service(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=RHYTHM_SERVICE_CLASS,
            paletteCategory="svc",
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
                    valueSchema=_core_features_schema(),
                )
            ],
            dataOutPorts=[
                F8DataPortSpec(
                    name="rhythmFeatures",
                    description="Rhythm feature payload.",
                    valueSchema=_rhythm_features_schema(),
                )
            ],
        ),
        AudioRhythmFeatureServiceNode,
        overwrite=True,
    )


def register_specs(registry: Registry) -> Registry:
    _register_core(registry)
    _register_rhythm(registry)
    return registry


def create_audiofeat_registry() -> RuntimeNodeRegistry:
    runtime_registry = create_runtime_node_registry()
    register_specs(Registry.wrap(runtime_registry))
    return runtime_registry


def shared_audiofeat_registry() -> RuntimeNodeRegistry:
    runtime_registry = shared_runtime_node_registry()
    register_specs(Registry.wrap(runtime_registry))
    return runtime_registry
