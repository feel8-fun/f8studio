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
_VIDEO_FRAME_METADATA_FIELDS: frozenset[str] = frozenset(
    {"schemaVersion", "format", "width", "height", "pitch", "frameId", "tsMs"}
)
_AUDIO_CHUNK_METADATA_FIELDS: frozenset[str] = frozenset(
    {"schemaVersion", "format", "sampleRate", "channels", "frames", "bytesPerFrame", "seq", "frameIndex", "tsMs"}
)


def _is_unset(value: object) -> bool:
    return isinstance(value, UnsetType)


def _payload_kind_from_value(value: object) -> F8DataPortPayloadKind:
    if isinstance(value, F8DataPortPayloadKind):
        return value
    text = str(value or "").strip()
    try:
        return F8DataPortPayloadKind(text)
    except ValueError:
        return F8DataPortPayloadKind.json


def _schema_comment_payload_kind(schema: F8DataTypeSchema) -> F8DataPortPayloadKind | None:
    comment = schema.field_comment
    if comment is None or _is_unset(comment):
        return None
    text = str(comment or "").strip()
    if text == "f8.payloadKind=video_frame":
        return F8DataPortPayloadKind.video_frame
    if text == "f8.payloadKind=audio_chunk":
        return F8DataPortPayloadKind.audio_chunk
    if text == "f8.payloadKind=bytes":
        return F8DataPortPayloadKind.bytes
    return None


def _schema_property_is_type(
    properties: dict[str, F8DataTypeSchema],
    *,
    name: str,
    schema_type_cls: type[object],
) -> bool:
    prop = properties.get(name)
    return isinstance(prop, schema_type_cls)


def _schema_string_enum_intersects(
    properties: dict[str, F8DataTypeSchema],
    *,
    name: str,
    allowed_values: tuple[str, ...],
) -> bool:
    prop = properties.get(name)
    if not isinstance(prop, F8StringTypeSchema):
        return False
    enum_values = prop.enum
    if enum_values is None or _is_unset(enum_values):
        return True
    allowed = set(allowed_values)
    return any(str(item) in allowed for item in list(enum_values or []))


def _legacy_schema_has_required_fields(schema: F8ComplexObjectTypeSchema, fields: frozenset[str]) -> bool:
    properties = schema.properties
    if not fields.issubset(set(properties.keys())):
        return False
    required = schema.required
    if required is None or _is_unset(required):
        return True
    return fields.issubset({str(item) for item in list(required or [])})


def _is_legacy_video_frame_metadata_schema(schema: F8DataTypeSchema) -> bool:
    if not isinstance(schema, F8ComplexObjectTypeSchema):
        return False
    if not _legacy_schema_has_required_fields(schema, _VIDEO_FRAME_METADATA_FIELDS):
        return False
    properties = schema.properties
    return (
        _schema_property_is_type(properties, name="schemaVersion", schema_type_cls=F8IntegerTypeSchema)
        and _schema_string_enum_intersects(properties, name="format", allowed_values=VIDEO_FRAME_FORMATS)
        and _schema_property_is_type(properties, name="width", schema_type_cls=F8IntegerTypeSchema)
        and _schema_property_is_type(properties, name="height", schema_type_cls=F8IntegerTypeSchema)
        and _schema_property_is_type(properties, name="pitch", schema_type_cls=F8IntegerTypeSchema)
        and _schema_property_is_type(properties, name="frameId", schema_type_cls=F8IntegerTypeSchema)
        and _schema_property_is_type(properties, name="tsMs", schema_type_cls=F8IntegerTypeSchema)
    )


def _is_legacy_audio_chunk_metadata_schema(schema: F8DataTypeSchema) -> bool:
    if not isinstance(schema, F8ComplexObjectTypeSchema):
        return False
    if not _legacy_schema_has_required_fields(schema, _AUDIO_CHUNK_METADATA_FIELDS):
        return False
    properties = schema.properties
    return (
        _schema_property_is_type(properties, name="schemaVersion", schema_type_cls=F8IntegerTypeSchema)
        and _schema_string_enum_intersects(properties, name="format", allowed_values=AUDIO_CHUNK_FORMATS)
        and _schema_property_is_type(properties, name="sampleRate", schema_type_cls=F8IntegerTypeSchema)
        and _schema_property_is_type(properties, name="channels", schema_type_cls=F8IntegerTypeSchema)
        and _schema_property_is_type(properties, name="frames", schema_type_cls=F8IntegerTypeSchema)
        and _schema_property_is_type(properties, name="bytesPerFrame", schema_type_cls=F8IntegerTypeSchema)
        and _schema_property_is_type(properties, name="seq", schema_type_cls=F8IntegerTypeSchema)
        and _schema_property_is_type(properties, name="frameIndex", schema_type_cls=F8IntegerTypeSchema)
        and _schema_property_is_type(properties, name="tsMs", schema_type_cls=F8IntegerTypeSchema)
    )


def _legacy_payload_kind_from_schema(schema: F8DataTypeSchema) -> F8DataPortPayloadKind:
    comment_kind = _schema_comment_payload_kind(schema)
    if comment_kind is not None:
        return comment_kind
    if _is_legacy_video_frame_metadata_schema(schema):
        return F8DataPortPayloadKind.video_frame
    if _is_legacy_audio_chunk_metadata_schema(schema):
        return F8DataPortPayloadKind.audio_chunk
    return F8DataPortPayloadKind.json


def data_port_payload_kind(port: F8DataPortSpec) -> F8DataPortPayloadKind:
    payload = port.payload
    if isinstance(payload, F8DataPayloadSpec):
        return _payload_kind_from_value(payload.kind)
    payload_kind = _payload_kind_from_value(port.payloadKind)
    if payload_kind != F8DataPortPayloadKind.json:
        return payload_kind
    return _legacy_payload_kind_from_schema(port.valueSchema)


def _delivery_from_value(value: object) -> F8DataPortDelivery:
    if isinstance(value, F8DataPortDelivery):
        return value
    text = str(value or "").strip()
    try:
        return F8DataPortDelivery(text)
    except ValueError:
        return F8DataPortDelivery.fifo


def data_port_stream_delivery(port: F8DataPortSpec) -> F8DataPortDelivery:
    stream = port.stream
    if isinstance(stream, F8DataStreamSpec):
        return _delivery_from_value(stream.delivery)
    payload_kind = data_port_payload_kind(port)
    if payload_kind in (F8DataPortPayloadKind.video_frame, F8DataPortPayloadKind.audio_chunk):
        return F8DataPortDelivery.latest
    return _delivery_from_value(port.delivery)


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
    if _is_unset(default_value):
        return None
    return None if default_value is None else default_value


def number_schema(
    *,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> F8NumberTypeSchema:
    kwargs: dict[str, object] = {}
    if default is not None:
        kwargs["default"] = default
    if minimum is not None:
        kwargs["minimum"] = minimum
    if maximum is not None:
        kwargs["maximum"] = maximum
    return F8NumberTypeSchema(**kwargs)


def string_schema(*, default: str | None = None, enum: list[str] | None = None) -> F8StringTypeSchema:
    kwargs: dict[str, object] = {}
    if default is not None:
        kwargs["default"] = default
    if enum is not None:
        kwargs["enum"] = enum
    return F8StringTypeSchema(**kwargs)


def integer_schema(
    *,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> F8IntegerTypeSchema:
    kwargs: dict[str, object] = {}
    if default is not None:
        kwargs["default"] = default
    if minimum is not None:
        kwargs["minimum"] = minimum
    if maximum is not None:
        kwargs["maximum"] = maximum
    return F8IntegerTypeSchema(**kwargs)


def boolean_schema(*, default: bool | None = None) -> F8BooleanTypeSchema:
    kwargs: dict[str, object] = {}
    if default is not None:
        kwargs["default"] = default
    return F8BooleanTypeSchema(**kwargs)


def array_schema(
    *,
    items: F8DataTypeSchema,
    default: list[object] | None = None,
) -> F8ArrayTypeSchema:
    kwargs: dict[str, object] = {"items": items}
    if default is not None:
        kwargs["default"] = list(default)
    return F8ArrayTypeSchema(**kwargs)


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


def video_frame_schema() -> F8ComplexObjectTypeSchema:
    """
    Backward-compatible alias for video_frame_metadata_schema().

    New data-port declarations should prefer video_frame_port().
    """

    return video_frame_metadata_schema()


def audio_chunk_schema() -> F8ComplexObjectTypeSchema:
    """
    Backward-compatible alias for audio_chunk_metadata_schema().

    New data-port declarations should prefer audio_chunk_port().
    """

    return audio_chunk_metadata_schema()


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
    required: bool = True,
    show_on_node: bool = True,
    delivery: F8DataPortDelivery = F8DataPortDelivery.fifo,
) -> F8DataPortSpec:
    return F8DataPortSpec(
        name=name,
        valueSchema=value_schema,
        payload=data_payload_spec(kind=F8DataPortPayloadKind.json, value_schema=value_schema),
        stream=data_stream_spec(delivery=delivery),
        description=UNSET if description is None else description,
        payloadKind=F8DataPortPayloadKind.json,
        delivery=delivery,
        required=bool(required),
        showOnNode=bool(show_on_node),
    )


def video_frame_port(
    *,
    name: str,
    description: str | None = None,
    required: bool = True,
    show_on_node: bool = True,
    formats: tuple[str, ...] | list[str] = VIDEO_FRAME_FORMATS,
) -> F8DataPortSpec:
    metadata_schema = video_frame_metadata_schema()
    return F8DataPortSpec(
        name=name,
        valueSchema=metadata_schema,
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
        payloadKind=F8DataPortPayloadKind.video_frame,
        delivery=F8DataPortDelivery.latest,
        required=bool(required),
        showOnNode=bool(show_on_node),
    )


def audio_chunk_port(
    *,
    name: str,
    description: str | None = None,
    required: bool = True,
    show_on_node: bool = True,
    formats: tuple[str, ...] | list[str] = AUDIO_CHUNK_FORMATS,
) -> F8DataPortSpec:
    metadata_schema = audio_chunk_metadata_schema()
    return F8DataPortSpec(
        name=name,
        valueSchema=metadata_schema,
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
        payloadKind=F8DataPortPayloadKind.audio_chunk,
        delivery=F8DataPortDelivery.latest,
        required=bool(required),
        showOnNode=bool(show_on_node),
    )


__all__ = [
    "AUDIO_CHUNK_FORMATS",
    "VIDEO_FRAME_FORMATS",
    "audio_chunk_metadata_schema",
    "audio_chunk_port",
    "audio_chunk_schema",
    "any_schema",
    "array_schema",
    "boolean_schema",
    "complex_object_schema",
    "data_port_payload_kind",
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
    "video_frame_schema",
]
