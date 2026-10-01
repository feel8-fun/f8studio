from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .format import EVENT_TYPE_DATA_SAMPLE, EVENT_TYPE_HEADER, EVENT_TYPE_STATE_CHANGE


@dataclass(frozen=True)
class RecordingHeader:
    format_version: int
    created_ts_ms: int
    data_ports: tuple[str, ...]
    state_fields: tuple[str, ...]


@dataclass(frozen=True)
class HeaderEvent:
    type: Literal["header"]
    header: RecordingHeader


@dataclass(frozen=True)
class DataSampleEvent:
    type: Literal["data_sample"]
    tick_ts_ms: int
    relative_offset_ms: int
    data: dict[str, Any]


@dataclass(frozen=True)
class StateChangeEvent:
    type: Literal["state_change"]
    state_ts_ms: int
    relative_offset_ms: int
    field: str
    value: Any


RecordingEvent = HeaderEvent | DataSampleEvent | StateChangeEvent


def header_event_to_dict(event: HeaderEvent) -> dict[str, Any]:
    return {
        "type": EVENT_TYPE_HEADER,
        "header": {
            "format_version": int(event.header.format_version),
            "created_ts_ms": int(event.header.created_ts_ms),
            "data_ports": list(event.header.data_ports),
            "state_fields": list(event.header.state_fields),
        },
    }


def data_sample_event_to_dict(event: DataSampleEvent) -> dict[str, Any]:
    return {
        "type": EVENT_TYPE_DATA_SAMPLE,
        "tick_ts_ms": int(event.tick_ts_ms),
        "relative_offset_ms": int(event.relative_offset_ms),
        "data": dict(event.data),
    }


def state_change_event_to_dict(event: StateChangeEvent) -> dict[str, Any]:
    return {
        "type": EVENT_TYPE_STATE_CHANGE,
        "state_ts_ms": int(event.state_ts_ms),
        "relative_offset_ms": int(event.relative_offset_ms),
        "field": str(event.field),
        "value": event.value,
    }


def event_to_dict(event: RecordingEvent) -> dict[str, Any]:
    if isinstance(event, HeaderEvent):
        return header_event_to_dict(event)
    if isinstance(event, DataSampleEvent):
        return data_sample_event_to_dict(event)
    return state_change_event_to_dict(event)


def parse_header(raw: Any) -> RecordingHeader:
    if not isinstance(raw, dict):
        raise ValueError("header payload must be an object")
    format_version = int(raw.get("format_version", 0))
    created_ts_ms = int(raw.get("created_ts_ms", 0))
    data_ports_raw = raw.get("data_ports")
    state_fields_raw = raw.get("state_fields")
    if not isinstance(data_ports_raw, list):
        raise ValueError("header.data_ports must be a list")
    if not isinstance(state_fields_raw, list):
        raise ValueError("header.state_fields must be a list")
    data_ports = tuple(str(item) for item in data_ports_raw if str(item).strip())
    state_fields = tuple(str(item) for item in state_fields_raw if str(item).strip())
    return RecordingHeader(
        format_version=format_version,
        created_ts_ms=created_ts_ms,
        data_ports=data_ports,
        state_fields=state_fields,
    )


def parse_event(raw: Any) -> RecordingEvent:
    if not isinstance(raw, dict):
        raise ValueError("recording event must be an object")
    event_type = str(raw.get("type") or "").strip()
    if event_type == EVENT_TYPE_HEADER:
        return HeaderEvent(type=EVENT_TYPE_HEADER, header=parse_header(raw.get("header")))
    if event_type == EVENT_TYPE_DATA_SAMPLE:
        payload = raw.get("data")
        if not isinstance(payload, dict):
            raise ValueError("data_sample.data must be an object")
        return DataSampleEvent(
            type=EVENT_TYPE_DATA_SAMPLE,
            tick_ts_ms=int(raw.get("tick_ts_ms", 0)),
            relative_offset_ms=int(raw.get("relative_offset_ms", 0)),
            data={str(k): v for k, v in payload.items()},
        )
    if event_type == EVENT_TYPE_STATE_CHANGE:
        return StateChangeEvent(
            type=EVENT_TYPE_STATE_CHANGE,
            state_ts_ms=int(raw.get("state_ts_ms", 0)),
            relative_offset_ms=int(raw.get("relative_offset_ms", 0)),
            field=str(raw.get("field") or ""),
            value=raw.get("value"),
        )
    raise ValueError(f"unknown recording event type: {event_type}")
