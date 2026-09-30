from __future__ import annotations

from msgspec import UNSET, UnsetType

from ..generated import (
    F8AnyTypeSchema,
    F8ArrayTypeSchema,
    F8BooleanTypeSchema,
    F8ComplexObjectTypeSchema,
    F8DataPayloadSpec,
    F8DataPortDelivery,
    F8DataPortPayloadKind,
    F8DataPortSpec,
    F8DataStreamCongestion,
    F8DataStreamPriority,
    F8DataStreamReliability,
    F8DataStreamSpec,
    F8DataTypeSchema,
    F8IntegerTypeSchema,
    F8NullTypeSchema,
    F8NumberTypeSchema,
    F8StringTypeSchema,
)

VIDEO_FRAME_FORMATS: tuple[str, ...] = ("bgra32", "bgr24", "flow2_f16", "scalar1_f32")
AUDIO_CHUNK_FORMATS: tuple[str, ...] = ("f32le",)


def data_port_payload_kind(port: F8DataPortSpec) -> F8DataPortPayloadKind:
    return port.payload.kind


def data_port_stream_delivery(port: F8DataPortSpec) -> F8DataPortDelivery:
    delivery = port.stream.delivery if isinstance(port.stream, F8DataStreamSpec) else UNSET
    return F8DataPortDelivery.fifo if isinstance(delivery, UnsetType) else delivery


def data_port_value_schema(port: F8DataPortSpec) -> F8DataTypeSchema:
    schema = port.payload.valueSchema if port.payload.kind == F8DataPortPayloadKind.json else port.payload.metadataSchema
    if isinstance(schema, UnsetType):
        raise ValueError(f"data port {port.name!r} is missing its payload schema")
    return schema


def schema_type(schema: F8DataTypeSchema) -> str:
    if isinstance(schema, F8StringTypeSchema):
        return "string"
    if isinstance(schema, F8NumberTypeSchema):
        return "number"
    if isinstance(schema, F8IntegerTypeSchema):
        return "integer"
    if isinstance(schema, F8BooleanTypeSchema):
        return "boolean"
    if isinstance(schema, F8NullTypeSchema):
        return "null"
    if isinstance(schema, F8ComplexObjectTypeSchema):
        return "object"
    if isinstance(schema, F8ArrayTypeSchema):
        return "array"
    if isinstance(schema, F8AnyTypeSchema):
        return "any"
    raise TypeError(f"unsupported schema type: {type(schema).__name__}")


def schema_default(schema: F8DataTypeSchema) -> object:
    default_value = schema.default
    if isinstance(default_value, UnsetType):
        return None
    return None if default_value is None else default_value


def number_schema(
    *,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> F8NumberTypeSchema:
    return F8NumberTypeSchema(
        default=default if default is not None else UNSET,
        minimum=minimum if minimum is not None else UNSET,
        maximum=maximum if maximum is not None else UNSET,
    )


def string_schema(*, default: str | None = None, enum: list[str] | None = None) -> F8StringTypeSchema:
    return F8StringTypeSchema(
        default=default if default is not None else UNSET,
        enum=enum if enum is not None else UNSET,
    )


def integer_schema(
    *,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> F8IntegerTypeSchema:
    return F8IntegerTypeSchema(
        default=default if default is not None else UNSET,
        minimum=minimum if minimum is not None else UNSET,
        maximum=maximum if maximum is not None else UNSET,
    )


def boolean_schema(*, default: bool | None = None) -> F8BooleanTypeSchema:
    return F8BooleanTypeSchema(
        default=default if default is not None else UNSET,
    )


def array_schema(
    *,
    items: F8DataTypeSchema,
    default: list[object] | None = None,
) -> F8ArrayTypeSchema:
    return F8ArrayTypeSchema(
        items=items,
        default=default if default is not None else UNSET,
    )


def any_schema() -> F8AnyTypeSchema:
    return F8AnyTypeSchema()


def complex_object_schema(
    *,
    properties: dict[str, F8DataTypeSchema],
) -> F8ComplexObjectTypeSchema:
    return F8ComplexObjectTypeSchema(properties=properties)


def video_frame_metadata_schema() -> F8ComplexObjectTypeSchema:
    """
    Metadata schema for binary video-frame data ports.

    The frame bytes are transported by the runtime stream layer, not by JSON.
    The object schema documents the decoded envelope metadata exposed by tools.
    """

    return F8ComplexObjectTypeSchema(
        title="F8 Video Frame Stream Metadata",
        description=(
            "Decoded metadata for a video_frame data stream. Frame bytes are carried by the runtime stream envelope, "
            "not by this JSON object."
        ),
        properties={
            "schemaVersion": integer_schema(default=2, minimum=2, maximum=2),
            "format": string_schema(default="bgra32", enum=list(VIDEO_FRAME_FORMATS)),
            "width": integer_schema(minimum=1),
            "height": integer_schema(minimum=1),
            "pitch": integer_schema(minimum=1),
            "frameId": integer_schema(minimum=1),
            "tsMs": integer_schema(minimum=0),
            "streamEpoch": string_schema(),
        },
        required=["schemaVersion", "format", "width", "height", "pitch", "frameId", "tsMs", "streamEpoch"],
        additionalProperties=False,
    )


def audio_chunk_metadata_schema() -> F8ComplexObjectTypeSchema:
    """
    Metadata schema for binary audio-chunk data ports.

    The PCM bytes are transported by the runtime stream layer, not by JSON.
    The object schema documents the decoded envelope metadata exposed by tools.
    """

    return F8ComplexObjectTypeSchema(
        title="F8 Audio Chunk Stream Metadata",
        description=(
            "Decoded metadata for an audio_chunk data stream. PCM bytes are carried by the runtime stream envelope, "
            "not by this JSON object."
        ),
        properties={
            "schemaVersion": integer_schema(default=1, minimum=1, maximum=1),
            "format": string_schema(default="f32le", enum=list(AUDIO_CHUNK_FORMATS)),
            "sampleRate": integer_schema(minimum=1),
            "channels": integer_schema(minimum=1),
            "frames": integer_schema(minimum=1),
            "bytesPerFrame": integer_schema(minimum=1),
            "seq": integer_schema(minimum=1),
            "frameIndex": integer_schema(minimum=0),
            "tsMs": integer_schema(minimum=0),
        },
        required=[
            "schemaVersion",
            "format",
            "sampleRate",
            "channels",
            "frames",
            "bytesPerFrame",
            "seq",
            "frameIndex",
            "tsMs",
        ],
        additionalProperties=False,
    )


def data_payload_spec(
    *,
    kind: F8DataPortPayloadKind,
    value_schema: F8DataTypeSchema | None = None,
    metadata_schema: F8DataTypeSchema | None = None,
    schema_version: int = 1,
    formats: tuple[str, ...] | list[str] = (),
) -> F8DataPayloadSpec:
    return F8DataPayloadSpec(
        kind=kind,
        valueSchema=UNSET if value_schema is None else value_schema,
        metadataSchema=UNSET if metadata_schema is None else metadata_schema,
        schemaVersion=int(schema_version),
        formats=list(formats),
    )


def data_stream_spec(
    *,
    delivery: F8DataPortDelivery = F8DataPortDelivery.fifo,
    reliability: F8DataStreamReliability = F8DataStreamReliability.best_effort,
    congestion: F8DataStreamCongestion = F8DataStreamCongestion.drop,
    priority: F8DataStreamPriority = F8DataStreamPriority.data,
) -> F8DataStreamSpec:
    return F8DataStreamSpec(
        delivery=delivery,
        reliability=reliability,
        congestion=congestion,
        priority=priority,
    )


def json_data_port(
    *,
    name: str,
    value_schema: F8DataTypeSchema,
    description: str | None = None,
    definition_protected: bool = True,
    show_on_node: bool = True,
    delivery: F8DataPortDelivery = F8DataPortDelivery.fifo,
) -> F8DataPortSpec:
    return F8DataPortSpec(
        name=name,
        payload=data_payload_spec(kind=F8DataPortPayloadKind.json, value_schema=value_schema),
        stream=data_stream_spec(delivery=delivery),
        description=UNSET if description is None else description,
        definitionProtected=bool(definition_protected),
        showOnNode=bool(show_on_node),
    )


def video_frame_port(
    *,
    name: str,
    description: str | None = None,
    definition_protected: bool = True,
    show_on_node: bool = True,
    formats: tuple[str, ...] | list[str] = VIDEO_FRAME_FORMATS,
) -> F8DataPortSpec:
    metadata_schema = video_frame_metadata_schema()
    return F8DataPortSpec(
        name=name,
        payload=data_payload_spec(
            kind=F8DataPortPayloadKind.video_frame,
            metadata_schema=metadata_schema,
            schema_version=2,
            formats=list(formats),
        ),
        stream=data_stream_spec(
            delivery=F8DataPortDelivery.latest,
            reliability=F8DataStreamReliability.best_effort,
            congestion=F8DataStreamCongestion.drop,
            priority=F8DataStreamPriority.real_time,
        ),
        description=UNSET if description is None else description,
        definitionProtected=bool(definition_protected),
        showOnNode=bool(show_on_node),
    )


def audio_chunk_port(
    *,
    name: str,
    description: str | None = None,
    definition_protected: bool = True,
    show_on_node: bool = True,
    formats: tuple[str, ...] | list[str] = AUDIO_CHUNK_FORMATS,
) -> F8DataPortSpec:
    metadata_schema = audio_chunk_metadata_schema()
    return F8DataPortSpec(
        name=name,
        payload=data_payload_spec(
            kind=F8DataPortPayloadKind.audio_chunk,
            metadata_schema=metadata_schema,
            formats=list(formats),
        ),
        stream=data_stream_spec(
            delivery=F8DataPortDelivery.latest,
            reliability=F8DataStreamReliability.best_effort,
            congestion=F8DataStreamCongestion.drop,
            priority=F8DataStreamPriority.real_time,
        ),
        description=UNSET if description is None else description,
        definitionProtected=bool(definition_protected),
        showOnNode=bool(show_on_node),
    )


__all__ = [
    "AUDIO_CHUNK_FORMATS",
    "VIDEO_FRAME_FORMATS",
    "audio_chunk_metadata_schema",
    "audio_chunk_port",
    "any_schema",
    "array_schema",
    "boolean_schema",
    "complex_object_schema",
    "data_port_payload_kind",
    "data_port_value_schema",
    "data_port_stream_delivery",
    "data_payload_spec",
    "data_stream_spec",
    "integer_schema",
    "json_data_port",
    "number_schema",
    "schema_default",
    "schema_type",
    "string_schema",
    "video_frame_metadata_schema",
    "video_frame_port",
]
