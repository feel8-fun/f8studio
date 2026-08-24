from __future__ import annotations

from f8pysdk.registry import Registry, RuntimeNodeRegistry, create_runtime_node_registry, shared_runtime_node_registry
from f8pysdk.specs import (
    F8DataPortSpec,
    F8ServiceSchemaVersion,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    array_schema,
    boolean_schema,
    integer_schema,
    string_schema,
)

from .constants import (
    DEFAULT_FUNCTIONAL_BONES,
    DEFAULT_POLL_INTERVAL_MS,
    DEFAULT_REFERENCE_PARTICIPANTS,
    DEFAULT_STALE_AFTER_MS,
    DEFAULT_TARGET_PARTICIPANTS,
    SERVICE_CLASS,
    SERVICE_VERSION,
)
from .source_node import FallenDollSourceNode


def _state_fields() -> list[F8StateSpec]:
    return [
        F8StateSpec(
            name="runtimeDir",
            label="Runtime Directory",
            description="Optional Fallen Doll runtime directory. Empty uses the Studio games directory.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=False,
            redactOnPublish=True,
        ),
        F8StateSpec(
            name="pollIntervalMs",
            label="Poll Interval (ms)",
            description="Interval for checking the game skeleton spool.",
            valueSchema=integer_schema(default=DEFAULT_POLL_INTERVAL_MS, minimum=10, maximum=1000),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="staleAfterMs",
            label="Stale After (ms)",
            description="Emit one empty safety frame after no HAnime skeleton update for this duration.",
            valueSchema=integer_schema(default=DEFAULT_STALE_AFTER_MS, minimum=50, maximum=10000),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="referenceRole",
            label="Reference Role",
            description="Participant role used as the interaction reference.",
            valueSchema=string_schema(default="male"),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="targetRole",
            label="Target Role",
            description="Participant role used as the moving target.",
            valueSchema=string_schema(default="female"),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="enabledReferenceParticipants",
            label="Reference Participants",
            description="Allowed stable participant keys, in addition to runtime priority metadata.",
            valueSchema=array_schema(items=string_schema(), default=DEFAULT_REFERENCE_PARTICIPANTS),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="enabledTargetParticipants",
            label="Target Participants",
            description="Allowed stable participant keys, in addition to runtime priority metadata.",
            valueSchema=array_schema(items=string_schema(), default=DEFAULT_TARGET_PARTICIPANTS),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="enabledReferenceBones",
            label="Reference Bones",
            description="Allowed functional reference bones. Runtime preferred-bone order wins.",
            valueSchema=array_schema(items=string_schema(), default=DEFAULT_FUNCTIONAL_BONES),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="enabledTargetBones",
            label="Target Bones",
            description="Allowed functional target bones. Runtime preferred-bone order wins.",
            valueSchema=array_schema(items=string_schema(), default=DEFAULT_FUNCTIONAL_BONES),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="resolvedPath",
            label="Resolved Path",
            description="Resolved skeleton spool path.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.ro,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="connected",
            label="Game Stream",
            description="True while fresh HAnime skeleton frames are arriving.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.ro,
            required=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="availableParticipants",
            label="Available Participants",
            description="Stable participant keys seen in the latest frame.",
            valueSchema=array_schema(items=string_schema()),
            access=F8StateAccess.ro,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="availableReferenceBones",
            label="Available Reference Bones",
            description="Bones on the selected reference participant.",
            valueSchema=array_schema(items=string_schema()),
            access=F8StateAccess.ro,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="availableTargetBones",
            label="Available Target Bones",
            description="Bones on the selected target participant.",
            valueSchema=array_schema(items=string_schema()),
            access=F8StateAccess.ro,
            required=True,
            showOnNode=False,
        ),
    ]


def register_specs(registry: Registry) -> Registry:
    registry.register_service(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=SERVICE_CLASS,
            paletteCategory="svc",
            version=SERVICE_VERSION,
            label="Fallen Doll Source",
            description="Reads HAnime skeleton frames exported by the Fallen Doll UE4SS mod.",
            tags=["game", "unreal", "skeleton", "fallen-doll"],
            rendererClass="default_svc",
            stateFields=_state_fields(),
            dataInPorts=[],
            dataOutPorts=[
                F8DataPortSpec(
                    name="skeletons",
                    description="Latest standard skeleton payloads.",
                    valueSchema=array_schema(items=any_schema()),
                ),
                F8DataPortSpec(
                    name="referenceSkeleton",
                    description="Selected reference participant skeleton.",
                    valueSchema=any_schema(),
                ),
                F8DataPortSpec(
                    name="targetSkeleton",
                    description="Selected target participant skeleton.",
                    valueSchema=any_schema(),
                ),
                F8DataPortSpec(
                    name="referenceBone",
                    description="Selected reference functional bone.",
                    valueSchema=any_schema(),
                ),
                F8DataPortSpec(
                    name="targetBone",
                    description="Selected target functional bone.",
                    valueSchema=any_schema(),
                ),
                F8DataPortSpec(
                    name="status",
                    description="Selection and HAnime identity status.",
                    valueSchema=any_schema(),
                ),
            ],
        ),
        FallenDollSourceNode,
        overwrite=True,
    )
    return registry


def create_fallen_doll_registry() -> RuntimeNodeRegistry:
    runtime_registry = create_runtime_node_registry()
    register_specs(Registry.wrap(runtime_registry))
    return runtime_registry


def shared_fallen_doll_registry() -> RuntimeNodeRegistry:
    runtime_registry = shared_runtime_node_registry()
    register_specs(Registry.wrap(runtime_registry))
    return runtime_registry
