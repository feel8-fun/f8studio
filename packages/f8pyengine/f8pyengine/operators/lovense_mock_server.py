from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from f8pysdk import (
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    boolean_schema,
    integer_schema,
    string_schema,
)
from f8pysdk.capabilities import ClosableNode, NodeBus
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry
from f8pysdk.time_utils import now_ms

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.lovense_mock_server"

_MAX_BODY_BYTES = 1024 * 1024
_MAX_HEADER_BYTES = 32 * 1024


@dataclass(frozen=True)
class _ServerConfig:
    bind_address: str
    port: int


def _now_iso(ts_ms: int) -> str:
    # ISO format with milliseconds, UTC "Z".
    from datetime import datetime, timezone

    dt = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _build_toy_map() -> dict[str, dict[str, Any]]:
    # Keep numeric-ish keys for stable order when consumers use Object.values(...) on the JSON object.
    return {
        "0": {
            "nickName": "Feel8 Lush",
            "name": "lush",
            "id": "MOCK_LUSH_0",
            "battery": 100,
            "version": "3",
            "status": "1",
        },
        "1": {
            "nickName": "Feel8 Solace",
            "name": "solace",
            "id": "MOCK_SOLACE_1",
            "battery": 100,
            "version": "1",
            "status": "1",
        },
    }


def _summarize_command(payload: dict[str, Any]) -> dict[str, Any]:
    cmd = payload.get("command")
    api_ver = payload.get("apiVer")

    if cmd == "Pattern":
        return {
            "type": "vibration_pattern",
            "toy": payload.get("toy"),
            "timeSec": payload.get("timeSec"),
            "strength": payload.get("strength"),
            "rule": payload.get("rule"),
            "apiVer": api_ver,
        }

    if cmd == "Function":
        action = str(payload.get("action") or "")
        thrusting: int | None = None
        depth: int | None = None
        if action.startswith("Thrusting:") and ",Depth:" in action:
            try:
                left, right = action.split(",Depth:", 1)
                thrusting = int(left.split("Thrusting:", 1)[1])
                depth = int(right)
            except Exception:
                thrusting = None
                depth = None

        if action == "Stop":
            typ = "stop"
        elif thrusting is not None and depth is not None:
            typ = "solace_thrusting"
        else:
            typ = "function"

        return {
            "type": typ,
            "toy": payload.get("toy"),
            "timeSec": payload.get("timeSec"),
            "action": action,
            "thrusting": thrusting,
            "depth": depth,
            "loopRunningSec": payload.get("loopRunningSec"),
            "loopPauseSec": payload.get("loopPauseSec"),
            "apiVer": api_ver,
        }

    if cmd == "GetToys":
        return {"type": "get_toys", "apiVer": api_ver}
    if cmd == "PatternV2":
        return {"type": "pattern_v2", "apiVer": api_ver}

    return {"type": "other", "command": cmd, "apiVer": api_ver}

def _unwrap_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, list, dict, tuple)):
        return value
    try:
        return value.root
    except Exception:
        return value


class LovenseMockServerRuntimeNode(OperatorNode, ClosableNode):
    """
    Mock Lovense Local API server (Mobile mode) for ingesting external commands.

    - POST /command (JSON)
    - Responds to GetToys with a minimal toy list
    - Captures all requests into a runtime-owned `event` state field

    This node is pure event-driven (no exec ports). It keeps the server running as
    long as the node instance stays registered in the ServiceHost, so rungraph
    redeploys that do not recreate the node won't drop connections.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})

        self._lock = asyncio.Lock()
        self._event_lock = asyncio.Lock()
        self._server: asyncio.AbstractServer | None = None
        self._cfg: _ServerConfig = _ServerConfig(
            bind_address=str(_unwrap_json_value(self._initial_state.get("bindAddress")) or "127.0.0.1"),
            port=self._parse_port(_unwrap_json_value(self._initial_state.get("port")), default=30010),
        )
        self._print_raw = self._parse_bool(_unwrap_json_value(self._initial_state.get("printRaw")), default=False)
        self._print_pretty = self._parse_bool(_unwrap_json_value(self._initial_state.get("printPretty")), default=False)

        self._seq = 0
        self._last_error: str | None = None

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        bus_like = bus if isinstance(bus, NodeBus) else None
        if bus_like is not None:
            try:
                if not bool(bus_like.active):
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._stop_server(), name=f"lovense_mock_server:deactivate:{self.node_id}")
                    return
            except Exception:
                pass
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._ensure_server(), name=f"lovense_mock_server:start:{self.node_id}")
        except Exception:
            pass

    async def close(self) -> None:
        await self._stop_server()

    async def on_lifecycle(self, active: bool, _meta: dict[str, Any]) -> None:
        if bool(active):
            await self._ensure_server()
        else:
            await self._stop_server()

    async def validate_state(
        self, field: str, value: Any, *, ts_ms: int | None = None, meta: dict[str, Any] | None = None
    ) -> Any:
        _ = ts_ms
        _ = meta
        value = _unwrap_json_value(value)
        name = str(field or "").strip()
        if name == "bindAddress":
            v = str(value or "").strip()
            if not v:
                raise ValueError("bindAddress must be non-empty")
            return v
        if name == "port":
            port = self._parse_port(value, default=30010)
            if port < 1 or port > 65535:
                raise ValueError("port must be 1..65535")
            return port
        if name == "printRaw":
            return self._parse_bool(value, default=False)
        if name == "printPretty":
            return self._parse_bool(value, default=False)
        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        _ = ts_ms
        name = str(field or "").strip()
        if name == "bindAddress":
            bind_address = str(_unwrap_json_value(value) or "").strip()
            if bind_address and bind_address != self._cfg.bind_address:
                self._cfg = _ServerConfig(bind_address=bind_address, port=self._cfg.port)
                await self._restart_server()
            return
        if name == "port":
            try:
                port = int(_unwrap_json_value(value))
            except Exception:
                return
            if port != self._cfg.port:
                self._cfg = _ServerConfig(bind_address=self._cfg.bind_address, port=port)
                await self._restart_server()
            return
        if name == "printRaw":
            self._print_raw = self._parse_bool(_unwrap_json_value(value), default=False)
            return
        if name == "printPretty":
            self._print_pretty = self._parse_bool(_unwrap_json_value(value), default=False)
            return

    async def _restart_server(self) -> None:
        await self._stop_server()
        await self._ensure_server()

    async def _ensure_server(self) -> None:
        async with self._lock:
            if self._server is not None:
                return
            cfg = self._cfg
            try:
                self._server = await asyncio.start_server(
                    self._handle_client,
                    host=str(cfg.bind_address),
                    port=int(cfg.port),
                    start_serving=True,
                )
                self._set_error(None)
            except OSError as exc:
                self._set_error(f"listen failed {cfg.bind_address}:{cfg.port}: {exc}")
                self._server = None
            except Exception as exc:
                self._set_error(f"listen failed {cfg.bind_address}:{cfg.port}: {exc}")
                self._server = None

        if self._server is None:
            await self._safe_set_state("listening", False)
            await self._safe_set_state("lastError", str(self._last_error or ""))
            return

        await self._safe_set_state("listening", True)
        await self._safe_set_state("lastError", "")

    async def _stop_server(self) -> None:
        server: asyncio.AbstractServer | None = None
        async with self._lock:
            server, self._server = self._server, None
        if server is None:
            await self._safe_set_state("listening", False)
            return
        try:
            server.close()
            await server.wait_closed()
        except Exception:
            pass
        await self._safe_set_state("listening", False)

    def _set_error(self, msg: str | None) -> None:
        self._last_error = msg
        if msg:
            print(f"[{self.node_id}:lovense_mock_server] {msg}")

    async def _safe_set_state(self, field: str, value: Any) -> None:
        try:
            await self.set_state(field, value)
        except Exception:
            return

    async def _safe_write_event(self, entry: dict[str, Any]) -> None:
        async with self._event_lock:
            self._seq += 1
            seq = int(self._seq)
        event = dict(entry)
        event["seq"] = seq
        # Include a changing field to avoid state value dedupe on repeats.
        event["eventId"] = f"{self.node_id}:{seq}"
        try:
            await self.set_state("event", event)
        except Exception:
            return

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await self._handle_client_inner(reader, writer)
        except Exception:
            try:
                await self._write_json(writer, status=500, obj={"ok": False, "error": "server_error"})
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle_client_inner(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        header_bytes = await self._read_until(reader, b"\r\n\r\n", limit=_MAX_HEADER_BYTES)
        if not header_bytes:
            return
        if not header_bytes.endswith(b"\r\n\r\n"):
            await self._write_json(writer, status=413, obj={"ok": False, "error": "headers_too_large"})
            return
        header_text = header_bytes.decode("utf-8", errors="replace")
        lines = header_text.split("\r\n")
        if not lines or not lines[0]:
            await self._write_json(writer, status=400, obj={"ok": False, "error": "bad_request"})
            return
        request_line = lines[0].strip()
        parts = request_line.split(" ")
        if len(parts) < 2:
            await self._write_json(writer, status=400, obj={"ok": False, "error": "bad_request"})
            return
        method = parts[0].upper()
        path = parts[1]

        headers: dict[str, str] = {}
        for raw_line in lines[1:]:
            if not raw_line or ":" not in raw_line:
                continue
            k, v = raw_line.split(":", 1)
            headers[k.strip().lower()] = v.strip()

        if method == "OPTIONS":
            await self._write_json(writer, status=204, obj=None)
            return

        if method != "POST" or path != "/command":
            await self._write_text(writer, status=404, text="Not Found")
            return

        content_length = 0
        if "content-length" in headers:
            try:
                content_length = int(headers["content-length"])
            except Exception:
                content_length = 0
        if content_length < 0 or content_length > _MAX_BODY_BYTES:
            await self._write_json(writer, status=413, obj={"ok": False, "error": "payload_too_large"})
            return

        body = b""
        if content_length > 0:
            body = await reader.readexactly(content_length)
        elif content_length == 0:
            body = b""
        else:
            await self._write_json(writer, status=400, obj={"ok": False, "error": "bad_request"})
            return

        try:
            payload_any = json.loads(body.decode("utf-8"))
        except Exception:
            entry = self._build_entry(raw="__invalid_json__", payload=None, remote=self._peer(writer))
            await self._safe_write_event(entry)
            await self._write_json(writer, status=400, obj={"ok": False, "error": "invalid_json"})
            return

        payload = payload_any if isinstance(payload_any, dict) else {"value": payload_any}
        entry = self._build_entry(raw=payload, payload=payload, remote=self._peer(writer))
        await self._safe_write_event(entry)

        if self._print_raw:
            if self._print_pretty:
                print(f"[{self.node_id}:lovense_mock_server] raw={json.dumps(payload, ensure_ascii=False, indent=2)}")
            else:
                print(f"[{self.node_id}:lovense_mock_server] raw={json.dumps(payload, ensure_ascii=False)}")
        else:
            print(f"[{self.node_id}:lovense_mock_server] {entry.get('summary')}")

        if payload.get("command") == "GetToys":
            toy_map = _build_toy_map()
            await self._write_json(writer, status=200, obj={"data": {"toys": json.dumps(toy_map, ensure_ascii=False)}})
            return

        await self._write_json(writer, status=200, obj={"ok": True})

    def _peer(self, writer: asyncio.StreamWriter) -> str:
        try:
            peer = writer.get_extra_info("peername")
        except Exception:
            peer = None
        if isinstance(peer, tuple) and peer:
            try:
                return str(peer[0])
            except Exception:
                return ""
        return ""

    def _build_entry(self, *, raw: Any, payload: dict[str, Any] | None, remote: str) -> dict[str, Any]:
        ts_ms = now_ms()
        summary: dict[str, Any] = {"type": "unknown"}
        if payload is not None:
            try:
                summary = _summarize_command(payload)
            except Exception:
                summary = {"type": "unknown"}
        return {
            "tsMs": int(ts_ms),
            "ts": _now_iso(int(ts_ms)),
            "remote": str(remote or ""),
            "path": "/command",
            "raw": raw,
            "summary": summary,
        }

    async def _read_until(self, reader: asyncio.StreamReader, marker: bytes, *, limit: int) -> bytes:
        try:
            data = await reader.readuntil(marker)
        except asyncio.LimitOverrunError:
            # Drain whatever was consumed so we can respond and close cleanly.
            try:
                _ = await reader.read(int(limit))
            except Exception:
                pass
            return b"__header_overrun__"
        except asyncio.IncompleteReadError as exc:
            data = bytes(exc.partial or b"")
        if not data:
            return b""
        if len(data) > int(limit):
            return data[: int(limit)]
        return bytes(data)

    async def _write_text(self, writer: asyncio.StreamWriter, *, status: int, text: str) -> None:
        body = (text or "").encode("utf-8")
        headers = self._response_headers(content_type="text/plain; charset=utf-8", content_length=len(body))
        status_line = f"HTTP/1.1 {status} {self._status_text(status)}\r\n"
        writer.write(status_line.encode("ascii") + headers + b"\r\n" + body)
        await writer.drain()

    async def _write_json(self, writer: asyncio.StreamWriter, *, status: int, obj: Any | None) -> None:
        if obj is None:
            body = b""
            content_type = "application/json; charset=utf-8"
        else:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        headers = self._response_headers(content_type=content_type, content_length=len(body))
        status_line = f"HTTP/1.1 {status} {self._status_text(status)}\r\n"
        writer.write(status_line.encode("ascii") + headers + b"\r\n" + body)
        await writer.drain()

    def _response_headers(self, *, content_type: str, content_length: int) -> bytes:
        lines = [
            f"Content-Type: {content_type}",
            f"Content-Length: {int(content_length)}",
            "Cache-Control: no-cache, no-store, max-age=0",
            "Access-Control-Allow-Origin: *",
            "Access-Control-Allow-Methods: POST, OPTIONS",
            "Access-Control-Allow-Headers: Content-Type, Accept",
            "Connection: close",
        ]
        return ("\r\n".join(lines) + "\r\n").encode("utf-8")

    def _status_text(self, status: int) -> str:
        if status == 200:
            return "OK"
        if status == 204:
            return "No Content"
        if status == 400:
            return "Bad Request"
        if status == 404:
            return "Not Found"
        if status == 413:
            return "Payload Too Large"
        if status == 500:
            return "Internal Server Error"
        return "OK"

    def _parse_bool(self, value: Any, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value or "").strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off", ""):
            return False
        return bool(default)

    def _parse_port(self, value: Any, *, default: int) -> int:
        try:
            v = int(value)
        except Exception:
            return int(default)
        if v < 1:
            return int(default)
        if v > 65535:
            return int(default)
        return int(v)


LovenseMockServerRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Lovense Mock Server",
    description="Event-driven input node that mocks the Lovense Local API and publishes received commands as state.",
    tags=["io", "lovense", "http", "server", "event"],
    stateFields=[
        F8StateSpec(
            name="bindAddress",
            label="Bind Address",
            description="Local address to bind (use 0.0.0.0 to accept other hosts on your LAN).",
            valueSchema=string_schema(default="127.0.0.1"),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="port",
            label="Port",
            description="HTTP port for the mock Lovense server.",
            valueSchema=integer_schema(default=30010, minimum=1, maximum=65535),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="printRaw",
            label="Print Raw JSON",
            description="Log full inbound command payloads (debug).",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="printPretty",
            label="Print Pretty JSON",
            description="Pretty-print JSON when printRaw is enabled (debug).",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="listening",
            label="Listening",
            description="True if the HTTP server is currently listening.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.ro,
            showOnNode=True,
        ),
        F8StateSpec(
            name="lastError",
            label="Last Error",
            description="Last server error (e.g. bind failure).",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.ro,
            showOnNode=True,
        ),
        F8StateSpec(
            name="event",
            label="Event",
            description="Latest received Lovense command (dict with seq/eventId/summary/raw).",
            valueSchema=any_schema(),
            access=F8StateAccess.ro,
            showOnNode=True,
        ),
    ],
    editableStateFields=False,
    editableDataInPorts=False,
    editableDataOutPorts=False,
    editableExecInPorts=False,
    editableExecOutPorts=False,
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return LovenseMockServerRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(LovenseMockServerRuntimeNode.SPEC, overwrite=True)
    return reg
