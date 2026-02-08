from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl

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
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


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
            "nickName": "Mock Lush",
            "name": "lush",
            "id": "MOCK_LUSH_0",
            "battery": 100,
            "fVersion": 0,
            "hVersion": 0,
            "version": "3",
            "connected": True,
            "status": "1",
            "domain": "127.0.0.1",
            "port": None,  # filled per-response
            "isHttps": False,
            "platform": "mock",
        },
        "1": {
            "nickName": "Mock Solace",
            "name": "solace",
            "id": "MOCK_SOLACE_1",
            "battery": 100,
            "fVersion": 0,
            "hVersion": 0,
            "version": "1",
            "connected": True,
            "status": "1",
            "domain": "127.0.0.1",
            "port": None,  # filled per-response
            "isHttps": False,
            "platform": "mock",
        },
    }


def _build_get_toys_response(*, toy_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    toys_string = json.dumps(toy_map, ensure_ascii=False, separators=(",", ":"))
    all_toys_by_id: dict[str, Any] = {}
    for toy in list(toy_map.values()):
        try:
            toy_id = str(toy.get("id") or "")
        except Exception:
            toy_id = ""
        if toy_id:
            all_toys_by_id[toy_id] = toy

    return {
        # Unity-friendly
        "code": 0,
        "type": "GetToys",
        "message": "OK",
        "data": {
            "toys": toys_string,
            "allToys": all_toys_by_id,
            "appType": "remote",
            "platform": "pc",
            "gameAppId": "",
        },
        # RPGM-friendly
        "ok": True,
        "data2": {"toys": toys_string, "toysMap": toy_map},
        # Convenience/debug
        "toys": toy_map,
    }


def _json_dumps_compact(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def _parse_body_text(body_text: str, content_type: str) -> Any:
    trimmed = str(body_text or "").strip()
    ct = str(content_type or "").lower()

    # Try JSON if it looks like JSON, regardless of content-type.
    if (
        "application/json" in ct
        or trimmed.startswith("{")
        or trimmed.startswith("[")
        or trimmed.startswith('"')
    ):
        return json.loads(trimmed)

    if "application/x-www-form-urlencoded" in ct:
        obj: dict[str, str] = {}
        for k, v in list(parse_qsl(trimmed, keep_blank_values=True)):
            obj[str(k)] = str(v)
        return obj

    return trimmed


def _normalize_payload(value: Any) -> dict[str, Any]:
    current: Any = value
    for _ in range(3):
        if isinstance(current, str):
            s = current.strip()
            looks_json = (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")) or (
                s.startswith('"') and s.endswith('"')
            )
            if looks_json:
                try:
                    current = json.loads(s)
                    continue
                except Exception:
                    return {"value": current}
            return {"value": current}

        if isinstance(current, list):
            if len(current) == 1 and isinstance(current[0], dict):
                current = current[0]
                continue
            return {"array": current}

        if isinstance(current, dict):
            data = current.get("data")
            if isinstance(data, str):
                s = data.strip()
                if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                    try:
                        current = json.loads(s)
                        continue
                    except Exception:
                        pass
            return current

        break

    return {"value": current}


def _summarize_command(payload: dict[str, Any]) -> dict[str, Any]:
    cmd_any = payload.get("command")
    if cmd_any is None:
        cmd_any = payload.get("cmd")
    if cmd_any is None:
        cmd_any = payload.get("type")
    if cmd_any is None:
        cmd_any = payload.get("request")
    if cmd_any is None:
        cmd_any = payload.get("method")

    cmd = str(cmd_any or "")
    if cmd == "ping":
        return {"type": "ping"}
    if cmd == "pong":
        return {"type": "pong"}

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
        all_v: int | None = None
        if action.startswith("Thrusting:") and ",Depth:" in action:
            try:
                left, right = action.split(",Depth:", 1)
                thrusting = int(left.split("Thrusting:", 1)[1])
                depth = int(right)
            except Exception:
                thrusting = None
                depth = None

        if action.startswith("All:"):
            try:
                all_v = int(action.split("All:", 1)[1])
            except Exception:
                all_v = None

        if thrusting is not None and depth is not None:
            typ = "solace_thrusting"
        elif all_v is not None:
            typ = "all_vibrate"
        elif action == "Stop":
            typ = "stop"
        else:
            typ = "function"

        return {
            "type": typ,
            "toy": payload.get("toy"),
            "timeSec": payload.get("timeSec"),
            "action": action,
            "thrusting": thrusting,
            "depth": depth,
            "all": all_v,
            "loopRunningSec": payload.get("loopRunningSec"),
            "loopPauseSec": payload.get("loopPauseSec"),
            "apiVer": api_ver,
        }

    if cmd == "GetToys":
        return {"type": "get_toys", "apiVer": api_ver if api_ver is not None else 1}
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
            bind_address=str(_unwrap_json_value(self._initial_state.get("bindAddress")) or "0.0.0.0"),
            port=self._parse_port(_unwrap_json_value(self._initial_state.get("port")), default=30010),
        )
        self._print_enabled = self._parse_bool(_unwrap_json_value(self._initial_state.get("printEnabled")), default=False)
        self._print_headers = self._parse_bool(_unwrap_json_value(self._initial_state.get("printHeaders")), default=False)
        self._print_body = self._parse_bool(_unwrap_json_value(self._initial_state.get("printBody")), default=False)
        self._print_responses = self._parse_bool(_unwrap_json_value(self._initial_state.get("printResponses")), default=False)
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
        if name == "printEnabled":
            return self._parse_bool(value, default=False)
        if name == "printHeaders":
            return self._parse_bool(value, default=False)
        if name == "printBody":
            return self._parse_bool(value, default=False)
        if name == "printResponses":
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
        if name == "printEnabled":
            self._print_enabled = self._parse_bool(_unwrap_json_value(value), default=False)
            return
        if name == "printHeaders":
            self._print_headers = self._parse_bool(_unwrap_json_value(value), default=False)
            return
        if name == "printBody":
            self._print_body = self._parse_bool(_unwrap_json_value(value), default=False)
            return
        if name == "printResponses":
            self._print_responses = self._parse_bool(_unwrap_json_value(value), default=False)
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

    async def _safe_write_event(self, entry: dict[str, Any]) -> dict[str, Any] | None:
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
            return None
        return event

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            # Basic keep-alive loop: handle multiple requests per TCP connection.
            while True:
                keep = await self._handle_one_request(reader, writer)
                if not keep:
                    break
        except Exception:
            try:
                await self._write_json(writer, status=500, obj={"ok": False, "error": "server_error"}, keep_alive=False)
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle_one_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> bool:
        header_bytes = await self._read_until(reader, b"\r\n\r\n", limit=_MAX_HEADER_BYTES)
        if header_bytes == b"__header_overrun__":
            await self._write_json(writer, status=413, obj={"ok": False, "error": "headers_too_large"}, keep_alive=False)
            return False
        if not header_bytes:
            return False
        if not header_bytes.endswith(b"\r\n\r\n"):
            await self._write_json(writer, status=413, obj={"ok": False, "error": "headers_too_large"}, keep_alive=False)
            return False
        header_text = header_bytes.decode("utf-8", errors="replace")
        lines = header_text.split("\r\n")
        if not lines or not lines[0]:
            await self._write_json(writer, status=400, obj={"ok": False, "error": "bad_request"}, keep_alive=False)
            return False
        request_line = lines[0].strip()
        parts = request_line.split(" ")
        if len(parts) < 2:
            await self._write_json(writer, status=400, obj={"ok": False, "error": "bad_request"}, keep_alive=False)
            return False
        method = parts[0].upper()
        path = parts[1]
        proto = parts[2] if len(parts) >= 3 else "HTTP/1.1"

        headers: dict[str, str] = {}
        for raw_line in lines[1:]:
            if not raw_line or ":" not in raw_line:
                continue
            k, v = raw_line.split(":", 1)
            headers[k.strip().lower()] = v.strip()

        path_s = str(path or "")
        conn = str(headers.get("connection") or "").strip().lower()
        # HTTP/1.1 defaults to keep-alive unless "Connection: close".
        keep_alive = (proto.upper().startswith("HTTP/1.1")) and conn != "close"
        if conn == "keep-alive":
            keep_alive = True

        # WebSocket (some SDKs use /v1).
        if (
            path_s == "/v1"
            and headers.get("upgrade", "").lower() == "websocket"
            and "sec-websocket-key" in headers
        ):
            if self._print_enabled and self._print_headers:
                try:
                    print(
                        f"[{self.node_id}:lovense_mock_server] WS upgrade {path_s} "
                        f"headers={json.dumps(self._pick_headers(headers), ensure_ascii=False, separators=(',', ':'))}"
                    )
                except Exception:
                    pass
            await self._handle_websocket_v1(reader, writer, headers)
            return False

        if method == "OPTIONS":
            await self._write_json(writer, status=204, obj=None, keep_alive=keep_alive)
            return keep_alive

        if method != "POST" or path != "/command":
            await self._write_text(writer, status=404, text="Not Found", keep_alive=keep_alive)
            return keep_alive

        content_type = headers.get("content-type", "")
        body_text = await self._read_body_text(reader, headers)
        if self._print_enabled and self._print_headers:
            try:
                picked = self._pick_headers(headers)
                print(
                    f"[{self.node_id}:lovense_mock_server] "
                    f"headers={json.dumps(picked, ensure_ascii=False, separators=(',', ':'))}"
                )
                if content_type and not picked.get("content-type"):
                    print(f"[{self.node_id}:lovense_mock_server] content-type={str(content_type)}")
            except Exception:
                pass
        if self._print_enabled and self._print_body:
            try:
                text = str(body_text or "")
                if len(text) > 2000:
                    head = text[:2000]
                    print(f"[{self.node_id}:lovense_mock_server] bodyText={head}...(truncated {len(text) - 2000})")
                else:
                    print(f"[{self.node_id}:lovense_mock_server] bodyText={text}")
            except Exception:
                pass
        try:
            raw_payload = _parse_body_text(body_text, content_type)
        except Exception as exc:
            ts_ms = int(now_ms())
            entry = {
                "tsMs": ts_ms,
                "ts": _now_iso(ts_ms),
                "remote": self._peer(writer),
                "path": str(path_s),
                "error": "parse_failed",
                "message": str(exc),
                "contentType": str(content_type),
                "bodyText": str(body_text),
            }
            await self._safe_write_event(entry)
            await self._write_json(writer, status=400, obj={"ok": False, "error": "parse_failed"}, keep_alive=keep_alive)
            return keep_alive

        normalized = _normalize_payload(raw_payload)
        summary = _summarize_command(normalized)

        ts_ms = int(now_ms())
        entry: dict[str, Any] = {
            "tsMs": ts_ms,
            "ts": _now_iso(ts_ms),
            "remote": self._peer(writer),
            "path": str(path_s),
            "raw": raw_payload,
            "summary": dict(summary),
            "contentType": str(content_type),
            "bodyText": str(body_text),
            "normalized": dict(normalized),
        }

        typ = str(summary.get("type") or "")

        # keep-alive traffic should not land in state.
        if typ in ("ping", "pong"):
            if typ == "ping":
                resp_obj = {"type": "pong"}
                if self._print_enabled and self._print_responses:
                    try:
                        print(f"[{self.node_id}:lovense_mock_server] resp status=200 body={_json_dumps_compact(resp_obj).decode('utf-8', errors='replace')}")
                    except Exception:
                        pass
                await self._write_json(writer, status=200, obj=resp_obj, keep_alive=keep_alive)
            else:
                resp_obj = {"ok": True}
                if self._print_enabled and self._print_responses:
                    try:
                        print(f"[{self.node_id}:lovense_mock_server] resp status=200 body={_json_dumps_compact(resp_obj).decode('utf-8', errors='replace')}")
                    except Exception:
                        pass
                await self._write_json(writer, status=200, obj=resp_obj, keep_alive=keep_alive)
            return keep_alive

        published_event: dict[str, Any] | None = None
        if typ in ("vibration_pattern", "solace_thrusting", "all_vibrate", "stop", "function", "pattern_v2"):
            published_event = await self._safe_write_event(entry)

        if self._print_enabled:
            if self._print_raw:
                event_to_print = published_event if isinstance(published_event, dict) else entry
                if self._print_pretty:
                    print(
                        f"[{self.node_id}:lovense_mock_server] "
                        f"event={json.dumps(event_to_print, ensure_ascii=False, indent=2)}"
                    )
                else:
                    print(
                        f"[{self.node_id}:lovense_mock_server] "
                        f"event={json.dumps(event_to_print, ensure_ascii=False, separators=(',', ':'))}"
                    )
            else:
                event_id = ""
                seq = ""
                if isinstance(published_event, dict):
                    event_id = str(published_event.get("eventId") or "")
                    seq = str(published_event.get("seq") or "")
                print(f"[{self.node_id}:lovense_mock_server] seq={seq} id={event_id} summary={entry.get('summary')}")

        if typ == "get_toys":
            toy_map = _build_toy_map()
            for toy in list(toy_map.values()):
                try:
                    toy["port"] = int(self._cfg.port)
                except Exception:
                    pass
            resp_obj = _build_get_toys_response(toy_map=toy_map)
            if self._print_enabled and self._print_responses:
                try:
                    dumped = _json_dumps_compact(resp_obj).decode("utf-8", errors="replace")
                    if len(dumped) > 2000:
                        print(f"[{self.node_id}:lovense_mock_server] resp status=200 body={dumped[:2000]}...(truncated {len(dumped) - 2000})")
                    else:
                        print(f"[{self.node_id}:lovense_mock_server] resp status=200 body={dumped}")
                except Exception:
                    pass
            await self._write_json(writer, status=200, obj=resp_obj, keep_alive=keep_alive)
            return keep_alive

        await self._write_json(writer, status=200, obj={"ok": True}, keep_alive=keep_alive)
        return keep_alive

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

    def _pick_headers(self, headers: dict[str, str]) -> dict[str, Any]:
        # Match scripts/lovense-mock-server.js: print a conservative subset.
        h = dict(headers or {})
        return {
            "content-type": h.get("content-type"),
            "content-length": h.get("content-length"),
            "user-agent": h.get("user-agent"),
            "accept": h.get("accept"),
            "accept-encoding": h.get("accept-encoding"),
            "connection": h.get("connection"),
            "upgrade": h.get("upgrade"),
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

    async def _read_body_text(self, reader: asyncio.StreamReader, headers: dict[str, str]) -> str:
        # Prefer Content-Length.
        content_length: int | None = None
        if "content-length" in headers:
            try:
                content_length = int(headers["content-length"])
            except Exception:
                content_length = None
        if content_length is not None:
            if content_length < 0 or content_length > _MAX_BODY_BYTES:
                return ""
            if content_length == 0:
                return ""
            body = await reader.readexactly(int(content_length))
            return body.decode("utf-8", errors="replace")

        # Minimal chunked support.
        if headers.get("transfer-encoding", "").lower() == "chunked":
            body = bytearray()
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    size_str = line.decode("ascii", errors="ignore").split(";", 1)[0].strip()
                    size = int(size_str, 16)
                except Exception:
                    break
                if size == 0:
                    _ = await reader.readline()
                    break
                if len(body) + size > _MAX_BODY_BYTES:
                    break
                chunk = await reader.readexactly(size)
                body.extend(chunk)
                _ = await reader.readexactly(2)  # CRLF
            return bytes(body).decode("utf-8", errors="replace")

        # Fallback: best-effort short read.
        body = bytearray()
        # Read until idle (handles clients that omit content-length but keep the socket open).
        while len(body) < _MAX_BODY_BYTES:
            try:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=0.05)
            except asyncio.TimeoutError:
                break
            except Exception:
                break
            if not chunk:
                break
            body.extend(chunk)
            # Stop once we have "something" and there's no immediate additional data.
            if len(body) >= _MAX_BODY_BYTES:
                break
        return bytes(body).decode("utf-8", errors="replace")

    async def _handle_websocket_v1(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, headers: dict[str, str]
    ) -> None:
        key = str(headers.get("sec-websocket-key") or "").strip()
        if not key:
            await self._write_text(writer, status=400, text="Bad Request", keep_alive=False)
            return
        accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode("utf-8")).digest()).decode("ascii")
        resp = "\r\n".join(
            [
                "HTTP/1.1 101 Switching Protocols",
                "Upgrade: websocket",
                "Connection: Upgrade",
                f"Sec-WebSocket-Accept: {accept}",
                "\r\n",
            ]
        )
        writer.write(resp.encode("utf-8"))
        await writer.drain()

        toy_map = _build_toy_map()
        for toy in list(toy_map.values()):
            try:
                toy["port"] = int(self._cfg.port)
            except Exception:
                pass
        await self._ws_send_text(writer, json.dumps({"type": "access-granted"}, ensure_ascii=False))
        await self._ws_send_text(writer, json.dumps({"type": "toy-list", "data": {"toys": toy_map}}, ensure_ascii=False))

        buf = bytearray()
        while True:
            chunk = await reader.read(4096)
            if not chunk:
                break
            buf.extend(chunk)
            frames, buf = self._ws_try_parse_frames(buf)
            for opcode, payload in frames:
                if opcode == 0x8:
                    await self._ws_send_close(writer)
                    return
                if opcode == 0x9:
                    await self._ws_send_pong(writer, payload)
                    continue
                if opcode != 0x1:
                    continue
                text = payload.decode("utf-8", errors="replace")
                try:
                    msg = json.loads(text)
                except Exception:
                    continue
                if not isinstance(msg, dict):
                    continue
                t = msg.get("type") or msg.get("command") or msg.get("cmd")
                t_s = str(t or "")
                if t_s == "ping":
                    await self._ws_send_text(writer, json.dumps({"type": "pong"}))
                if t_s == "access":
                    await self._ws_send_text(writer, json.dumps({"type": "access-granted"}))

    def _ws_try_parse_frames(self, data: bytearray) -> tuple[list[tuple[int, bytes]], bytearray]:
        buf = bytes(data)
        offset = 0
        out: list[tuple[int, bytes]] = []
        while len(buf) - offset >= 2:
            b0 = buf[offset]
            b1 = buf[offset + 1]
            fin = (b0 & 0x80) != 0
            opcode = b0 & 0x0F
            masked = (b1 & 0x80) != 0
            ln = b1 & 0x7F
            pos = offset + 2
            if ln == 126:
                if len(buf) - pos < 2:
                    break
                ln = int.from_bytes(buf[pos : pos + 2], "big")
                pos += 2
            elif ln == 127:
                if len(buf) - pos < 8:
                    break
                hi = int.from_bytes(buf[pos : pos + 4], "big")
                lo = int.from_bytes(buf[pos + 4 : pos + 8], "big")
                if hi != 0:
                    break
                ln = lo
                pos += 8
            if masked:
                if len(buf) - pos < 4:
                    break
                mask = buf[pos : pos + 4]
                pos += 4
            else:
                mask = None
            if len(buf) - pos < ln:
                break
            payload = buf[pos : pos + ln]
            pos += ln
            offset = pos
            if mask is not None:
                payload = bytes(payload[i] ^ mask[i % 4] for i in range(len(payload)))
            if not fin:
                continue
            out.append((int(opcode), bytes(payload)))
        remain = bytearray(buf[offset:])
        return out, remain

    async def _ws_send_text(self, writer: asyncio.StreamWriter, text: str) -> None:
        payload = str(text).encode("utf-8")
        header = bytearray()
        header.append(0x81)
        ln = len(payload)
        if ln < 126:
            header.append(ln)
        elif ln < 65536:
            header.append(126)
            header.extend(int(ln).to_bytes(2, "big"))
        else:
            header.append(127)
            header.extend((0).to_bytes(4, "big"))
            header.extend(int(ln).to_bytes(4, "big"))
        writer.write(bytes(header) + payload)
        await writer.drain()

    async def _ws_send_pong(self, writer: asyncio.StreamWriter, payload: bytes) -> None:
        p = bytes(payload or b"")
        header = bytearray([0x8A, len(p)])
        writer.write(bytes(header) + p)
        await writer.drain()

    async def _ws_send_close(self, writer: asyncio.StreamWriter) -> None:
        writer.write(b"\x88\x00")
        await writer.drain()

    async def _write_text(self, writer: asyncio.StreamWriter, *, status: int, text: str, keep_alive: bool) -> None:
        body = (text or "").encode("utf-8")
        headers = self._response_headers(
            content_type="text/plain; charset=utf-8", content_length=len(body), keep_alive=keep_alive
        )
        status_line = f"HTTP/1.1 {status} {self._status_text(status)}\r\n"
        writer.write(status_line.encode("ascii") + headers + b"\r\n" + body)
        await writer.drain()

    async def _write_json(
        self, writer: asyncio.StreamWriter, *, status: int, obj: Any | None, keep_alive: bool
    ) -> None:
        if obj is None:
            body = b""
            content_type = "application/json; charset=utf-8"
        else:
            body = _json_dumps_compact(obj)
            content_type = "application/json; charset=utf-8"
        headers = self._response_headers(content_type=content_type, content_length=len(body), keep_alive=keep_alive)
        status_line = f"HTTP/1.1 {status} {self._status_text(status)}\r\n"
        writer.write(status_line.encode("ascii") + headers + b"\r\n" + body)
        await writer.drain()

    def _response_headers(self, *, content_type: str, content_length: int, keep_alive: bool) -> bytes:
        lines = [
            f"Content-Type: {content_type}",
            f"Content-Length: {int(content_length)}",
            "Cache-Control: no-cache, no-store, max-age=0",
            "Access-Control-Allow-Origin: *",
            "Access-Control-Allow-Methods: POST, OPTIONS",
            "Access-Control-Allow-Headers: Content-Type, Accept",
        ]
        # Match Node's behavior more closely: do not force keep-alive via header.
        # Only explicitly request close when we intend to close.
        if not keep_alive:
            lines.append("Connection: close")
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
            valueSchema=string_schema(default="0.0.0.0"),
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
            name="printEnabled",
            label="Print Incoming",
            description="Print received commands to stdout (debug).",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="printHeaders",
            label="Print Headers",
            description="Print selected request headers to stdout (debug).",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="printBody",
            label="Print Body",
            description="Print request body text to stdout (debug).",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="printResponses",
            label="Print Responses",
            description="Print response JSON bodies to stdout (debug).",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            showOnNode=False,
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
