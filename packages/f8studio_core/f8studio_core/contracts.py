from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias


API_PROTOCOL_VERSION = "f8studio-api/1"
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True)
class HealthStatus:
    status: Literal["ok"]
    service: str
    version: str
    protocol_version: str
    server_epoch: str

    def to_json_object(self) -> JsonObject:
        return asdict(self)


@dataclass(frozen=True)
class ServerCapabilities:
    graph_editing: bool
    runtime_control: bool
    web_assets: bool
    web_rtc_video: bool
    web_rtc_audio: bool
    three_d: bool
    agent_tools: bool

    def to_json_object(self) -> JsonObject:
        return asdict(self)
