from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import platform
import sqlite3
import socket
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from threading import RLock
from typing import Any, Literal, cast
from uuid import uuid4

import msgspec
from f8pysdk.motion import SkeletonPacketDecodeError, decode_skeleton_datagram
from f8pysdk.specs import F8JsonValue

from .native_hotkeys import (
    NativeHotkeyBackend,
    NativeHotkeyBinding,
    NativeHotkeyRegistrationError,
    NativeHotkeyUnsupportedError,
    create_native_hotkey_backend,
    parse_native_hotkey,
)


DEFAULT_SKELETON_UDP_PORT = 39540
logger = logging.getLogger(__name__)


class LocalCapability(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    capability: str
    status: Literal["available", "unavailable", "unverified"]
    backend: str
    reason: str = ""


class SerialPortInfo(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    device: str
    description: str
    hardware_id: str


class DetectModdingTargetRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    target_path: str


class PreviewUnityInstallRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    target_path: str
    exporter: Literal["auto", "skeleton", "live2d"] = "auto"
    udp_port: int = DEFAULT_SKELETON_UDP_PORT
    offline: bool = True
    force_reinstall: bool = False
    skip_exporter: bool = False


class ApplyUnityInstallRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    plan_id: str
    confirm: bool


class UnityInstallPlan(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    plan_id: str
    target_path: str
    actions: tuple[str, ...]
    blocking_errors: tuple[str, ...]
    files_to_write: tuple[str, ...]
    files_to_preserve: tuple[str, ...]
    graph_build_plan: F8JsonValue
    raw: F8JsonValue


class VerifySkeletonUdpRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    bind_address: str = "127.0.0.1"
    port: int = DEFAULT_SKELETON_UDP_PORT
    timeout_ms: int = 2000
    minimum_frames: int = 1


class SkeletonUdpVerification(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    bind_address: str
    port: int
    packet_count: int
    decoded_frame_count: int
    model_names: tuple[str, ...]
    decoder_errors: tuple[str, ...]
    verified: bool


class HotkeyBinding(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    binding_id: str
    accelerator: str
    project_id: str
    node_id: str
    field: str
    status: Literal["configured", "registered", "disabled", "error"] = "configured"
    message: str = ""


class RegisterHotkeyRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    accelerator: str
    project_id: str
    node_id: str
    field: str
    binding_id: str | None = None


def _json(value: object) -> F8JsonValue:
    return cast(F8JsonValue, msgspec.to_builtins(value, str_keys=True))


def skeleton_graph_build_plan(port: int) -> F8JsonValue:
    return {
        "schemaVersion": "f8modding-graph-plan/1",
        "summary": "UDP skeleton preview",
        "nodes": [
            {"operatorClass": "f8.udp_in", "nodeId": "modding_udp_in", "stateValues": {"bindAddress": "127.0.0.1", "port": port}},
            {"operatorClass": "f8.skeleton_decoder", "nodeId": "modding_skeleton_decoder", "stateValues": {}},
            {"operatorClass": "f8.viz.three_d", "nodeId": "modding_viz_3d", "stateValues": {}},
        ],
        "connections": [
            {"fromNodeId": "modding_udp_in", "fromPort": "packet", "toNodeId": "modding_skeleton_decoder", "toPort": "packet"},
            {"fromNodeId": "modding_skeleton_decoder", "fromPort": "skeletons", "toNodeId": "modding_viz_3d", "toPort": "skeletons"},
        ],
        "safety": {"physicalOutputArmed": False, "watchdogTimeoutMs": 250},
    }


HotkeyActivation = Callable[[HotkeyBinding], Awaitable[None]]
HotkeyValidator = Callable[[HotkeyBinding], None]


class LocalIntegrationService:
    def __init__(
        self,
        *,
        database_path: Path | None = None,
        hotkey_backend: NativeHotkeyBackend | None = None,
        hotkey_activation: HotkeyActivation | None = None,
        hotkey_validator: HotkeyValidator | None = None,
    ) -> None:
        self._unity_plans: dict[str, tuple[PreviewUnityInstallRequest, dict[str, object]]] = {}
        self._database_path = database_path.resolve() if database_path is not None else None
        self._hotkeys = self._load_hotkeys()
        self._hotkey_status: dict[str, tuple[Literal["registered", "disabled", "error"], str]] = {}
        self._hotkey_backend = hotkey_backend
        self._hotkey_backend_injected = hotkey_backend is not None
        self._hotkey_backend_error = ""
        self._hotkey_activation = hotkey_activation
        self._hotkey_validator = hotkey_validator
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._activation_futures: set[concurrent.futures.Future[None]] = set()
        self._hotkey_lock = RLock()

    async def start(self) -> None:
        self._event_loop = asyncio.get_running_loop()
        if self._hotkey_backend is None and not self._hotkey_backend_injected:
            try:
                self._hotkey_backend = await asyncio.to_thread(
                    create_native_hotkey_backend,
                    self._on_native_hotkey,
                )
            except NativeHotkeyUnsupportedError as exc:
                self._hotkey_backend_error = str(exc)
                logger.info("native global hotkeys unavailable: %s", exc)
            except Exception as exc:
                self._hotkey_backend_error = f"{type(exc).__name__}: {exc}"
                logger.exception("native global hotkey backend initialization failed", exc_info=exc)
        await asyncio.to_thread(self._refresh_hotkey_backend)

    async def close(self) -> None:
        backend = self._hotkey_backend
        self._hotkey_backend = None
        if backend is not None:
            try:
                await asyncio.to_thread(backend.close)
            except Exception as exc:
                logger.exception("native global hotkey backend close failed", exc_info=exc)
        for future in tuple(self._activation_futures):
            future.cancel()
        self._activation_futures.clear()
        self._event_loop = None

    def capabilities(self) -> tuple[LocalCapability, ...]:
        unity_available = self._unity_available()
        serial_available = self._serial_available()
        native_backend = self._hotkey_backend
        system = platform.system()
        return (
            LocalCapability(
                capability="unity_modding",
                status="available" if unity_available else "unavailable",
                backend="f8unitymods_setup",
                reason="" if unity_available else "f8unitymods_setup is not installed in this environment",
            ),
            LocalCapability(
                capability="unreal_modding",
                status="unverified",
                backend="typed_contract",
                reason="No repository-owned UE4SS installer is available; detection remains a release gate.",
            ),
            LocalCapability(
                capability="vam_modding",
                status="unverified",
                backend="typed_contract",
                reason="VaM package installation requires a Windows target and remains a release gate.",
            ),
            LocalCapability(
                capability="serial_ports",
                status="available" if serial_available else "unavailable",
                backend="pyserial",
                reason="" if serial_available else "pyserial is not installed in this environment",
            ),
            LocalCapability(
                capability="udp_skeleton_verify",
                status="available",
                backend="f8pysdk.motion",
            ),
            LocalCapability(
                capability="global_hotkeys",
                status="available" if native_backend is not None else "unavailable",
                backend="win32" if system == "Windows" else "x11" if system == "Linux" else "unsupported",
                reason=self._hotkey_backend_error,
            ),
            LocalCapability(
                capability="managed_processes",
                status="available",
                backend="catalog_allowlist",
                reason="Only catalog-declared service executables may be started.",
            ),
        )

    def serial_ports(self) -> tuple[SerialPortInfo, ...]:
        try:
            from serial.tools import list_ports
        except ImportError as exc:
            raise RuntimeError("pyserial is unavailable in the Web Studio runtime environment") from exc
        return tuple(
            SerialPortInfo(
                device=str(port.device),
                description=str(port.description or ""),
                hardware_id=str(port.hwid or ""),
            )
            for port in sorted(list_ports.comports(), key=lambda item: str(item.device))
        )

    def detect_modding_target(self, request: DetectModdingTargetRequest) -> F8JsonValue:
        target = Path(request.target_path).expanduser().resolve()
        if not target.exists():
            raise FileNotFoundError(f"modding target not found: {target}")
        suffix = target.suffix.lower()
        if suffix == ".exe" or any(target.glob("*_Data")):
            engine = "unity"
        elif suffix == ".uproject" or any(target.glob("Engine/Binaries/*")):
            engine = "unreal"
        elif target.name.lower() in {"vam.exe", "virt-a-mate"} or (target / "AddonPackages").is_dir():
            engine = "vam"
        else:
            engine = "unknown"
        return {
            "targetPath": str(target),
            "engine": engine,
            "supported": engine == "unity" and self._unity_available(),
            "reason": "" if engine == "unity" else "Only the repository-owned Unity installer can currently apply changes.",
        }

    def preview_unity_install(self, request: PreviewUnityInstallRequest) -> UnityInstallPlan:
        if not (1 <= request.udp_port <= 65535):
            raise ValueError("udpPort must be between 1 and 65535")
        try:
            from f8unitymods_setup import game_setup
            from f8unitymods_setup.common import load_setup_config
        except ImportError as exc:
            raise RuntimeError("f8unitymods_setup is unavailable in the Web Studio runtime environment") from exc
        target = Path(request.target_path).expanduser().resolve()
        if not target.exists():
            raise FileNotFoundError(f"Unity target not found: {target}")
        raw = cast(dict[str, object], game_setup.run_diagnose(
            target=str(target),
            config=load_setup_config(),
            exporter=request.exporter,
            prefer_local_configs=True,
            allow_remote_configs=not request.offline,
            refresh_remote_cache=False,
            release_tag="",
            force_reinstall=request.force_reinstall,
            skip_exporter=request.skip_exporter,
            rue=False,
            cue=False,
            config_manager=False,
            uud=False,
            offline=request.offline,
        ))
        raw_plan = raw.get("plan")
        plan_payload = cast(dict[str, object], raw_plan) if isinstance(raw_plan, dict) else {}
        direct_actions = raw.get("actions")
        raw_actions: object = cast(list[object], direct_actions) if isinstance(direct_actions, list) else plan_payload.get("actions", [])
        raw_blocking = plan_payload.get("blocking_errors", [])
        actions = (
            tuple(str(value) for value in cast(list[object], raw_actions) if str(value).strip())
            if isinstance(raw_actions, list)
            else ()
        )
        blocking = (
            tuple(str(value) for value in cast(list[object], raw_blocking) if str(value).strip())
            if isinstance(raw_blocking, list)
            else ()
        )
        plan_id = uuid4().hex
        self._unity_plans[plan_id] = (request, raw)
        writes = tuple(self._action_write_hints(action) for action in actions)
        flattened_writes = tuple(dict.fromkeys(path for group in writes for path in group))
        preserves = ("Existing unmanaged BepInEx configuration is preserved unless the preview identifies a managed file.",)
        return UnityInstallPlan(
            plan_id=plan_id,
            target_path=str(target),
            actions=actions,
            blocking_errors=blocking,
            files_to_write=flattened_writes,
            files_to_preserve=preserves,
            graph_build_plan=skeleton_graph_build_plan(request.udp_port),
            raw=_json(raw),
        )

    def apply_unity_install(self, request: ApplyUnityInstallRequest) -> F8JsonValue:
        if not request.confirm:
            raise ValueError("Unity installation requires confirm=true after reviewing the plan")
        stored = self._unity_plans.pop(request.plan_id, None)
        if stored is None:
            raise FileNotFoundError(f"Unity install plan not found or already used: {request.plan_id}")
        preview, raw = stored
        raw_plan = raw.get("plan")
        plan_payload = cast(dict[str, object], raw_plan) if isinstance(raw_plan, dict) else {}
        blocking = plan_payload.get("blocking_errors", [])
        if isinstance(blocking, list) and blocking:
            raise ValueError("Unity install plan has blocking errors")
        try:
            from f8unitymods_setup import game_setup
            from f8unitymods_setup.common import load_setup_config
        except ImportError as exc:
            raise RuntimeError("f8unitymods_setup is unavailable in the Web Studio runtime environment") from exc
        result = game_setup.run_install(
            target=str(Path(preview.target_path).expanduser().resolve()),
            config=load_setup_config(),
            exporter=preview.exporter,
            prefer_local_configs=True,
            allow_remote_configs=not preview.offline,
            refresh_remote_cache=False,
            release_tag="",
            force_reinstall=preview.force_reinstall,
            rue=False,
            cue=False,
            config_manager=False,
            uud=False,
            skip_exporter=preview.skip_exporter,
            offline=preview.offline,
            interaction_meta={"interaction_used": False, "target_prompted": False, "exporter_prompted": False},
        )
        return _json(result)

    async def verify_skeleton_udp(self, request: VerifySkeletonUdpRequest) -> SkeletonUdpVerification:
        if not (1 <= request.port <= 65535):
            raise ValueError("UDP port must be between 1 and 65535")
        if not (100 <= request.timeout_ms <= 30_000):
            raise ValueError("timeoutMs must be between 100 and 30000")
        if not (1 <= request.minimum_frames <= 100):
            raise ValueError("minimumFrames must be between 1 and 100")
        return await asyncio.to_thread(self._verify_skeleton_udp_blocking, request)

    def list_hotkeys(self, project_id: str | None = None) -> tuple[HotkeyBinding, ...]:
        with self._hotkey_lock:
            return tuple(
                self._binding_with_status(binding)
                for binding in self._hotkeys.values()
                if project_id is None or binding.project_id == project_id
            )

    def register_hotkey(self, request: RegisterHotkeyRequest) -> HotkeyBinding:
        with self._hotkey_lock:
            accelerator = parse_native_hotkey(request.accelerator).display_text
            binding_id = request.binding_id or uuid4().hex
            conflict = next(
                (
                    binding
                    for binding in self._hotkeys.values()
                    if binding.accelerator == accelerator and binding.binding_id != binding_id
                ),
                None,
            )
            if conflict is not None:
                raise ValueError(f"hotkey accelerator is already registered: {accelerator}")
            binding = HotkeyBinding(
                binding_id=binding_id,
                accelerator=accelerator,
                project_id=request.project_id,
                node_id=request.node_id,
                field=request.field,
            )
            if self._hotkey_validator is not None:
                self._hotkey_validator(binding)
            self._save_hotkey(binding)
            self._hotkeys[binding.binding_id] = binding
            self._refresh_hotkey_backend()
            return self._binding_with_status(binding)

    def unregister_hotkey(self, binding_id: str) -> None:
        with self._hotkey_lock:
            if binding_id not in self._hotkeys:
                raise FileNotFoundError(f"hotkey binding not found: {binding_id}")
            self._delete_hotkey(binding_id)
            self._hotkeys.pop(binding_id)
            self._refresh_hotkey_backend()

    def refresh_hotkeys(self) -> None:
        with self._hotkey_lock:
            self._refresh_hotkey_backend()

    def forget_project_hotkeys(self, project_id: str) -> None:
        with self._hotkey_lock:
            self._hotkeys = {
                binding_id: binding
                for binding_id, binding in self._hotkeys.items()
                if binding.project_id != project_id
            }
            self._refresh_hotkey_backend()

    @staticmethod
    def _verify_skeleton_udp_blocking(request: VerifySkeletonUdpRequest) -> SkeletonUdpVerification:
        packet_count = 0
        decoded: list[dict[str, Any]] = []
        errors: list[str] = []
        names: set[str] = set()
        stop_at = time.monotonic() + request.timeout_ms / 1000.0
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as listener:
            listener.settimeout(min(0.2, request.timeout_ms / 1000.0))
            listener.bind((request.bind_address, request.port))
            while time.monotonic() < stop_at and len(decoded) < request.minimum_frames:
                try:
                    packet, _address = listener.recvfrom(1_048_576)
                except TimeoutError:
                    continue
                packet_count += 1
                try:
                    frame = decode_skeleton_datagram(packet)
                except (SkeletonPacketDecodeError, ValueError, UnicodeDecodeError) as exc:
                    if len(errors) < 8:
                        errors.append(f"{type(exc).__name__}: {exc}")
                    continue
                model_name = frame.get("modelName")
                bones = frame.get("bones")
                if not isinstance(model_name, str) or not model_name or not isinstance(bones, list):
                    if len(errors) < 8:
                        errors.append("decoded payload is missing modelName or bones")
                    continue
                decoded.append(frame)
                names.add(model_name)
        return SkeletonUdpVerification(
            bind_address=request.bind_address,
            port=request.port,
            packet_count=packet_count,
            decoded_frame_count=len(decoded),
            model_names=tuple(sorted(names)),
            decoder_errors=tuple(errors),
            verified=len(decoded) >= request.minimum_frames,
        )

    def _refresh_hotkey_backend(self) -> None:
        backend = self._hotkey_backend
        self._hotkey_status.clear()
        if backend is None:
            message = self._hotkey_backend_error or "Native global hotkey backend is not running"
            self._hotkey_status.update(
                (binding_id, ("disabled", message)) for binding_id in self._hotkeys
            )
            return
        try:
            backend.unregister_all()
        except Exception as exc:
            logger.exception("failed to clear native global hotkeys before refresh", exc_info=exc)
            message = f"{type(exc).__name__}: {exc}"
            self._hotkey_status.update((binding_id, ("error", message)) for binding_id in self._hotkeys)
            return
        for binding in self._hotkeys.values():
            if self._hotkey_validator is not None:
                try:
                    self._hotkey_validator(binding)
                except (FileNotFoundError, ValueError) as exc:
                    self._hotkey_status[binding.binding_id] = ("error", str(exc))
                    continue
            try:
                backend.register_hotkey(
                    NativeHotkeyBinding(
                        binding_id=binding.binding_id,
                        spec=parse_native_hotkey(binding.accelerator),
                    )
                )
            except NativeHotkeyRegistrationError as exc:
                self._hotkey_status[binding.binding_id] = ("error", str(exc))
                logger.warning(
                    "native global hotkey registration failed binding_id=%s accelerator=%s",
                    binding.binding_id,
                    binding.accelerator,
                    exc_info=exc,
                )
            else:
                self._hotkey_status[binding.binding_id] = ("registered", "")

    def _binding_with_status(self, binding: HotkeyBinding) -> HotkeyBinding:
        status, message = self._hotkey_status.get(binding.binding_id, ("configured", ""))
        return msgspec.structs.replace(binding, status=status, message=message)

    def _on_native_hotkey(self, binding_id: str) -> None:
        with self._hotkey_lock:
            binding = self._hotkeys.get(binding_id)
        loop = self._event_loop
        activation = self._hotkey_activation
        if binding is None or loop is None or activation is None or loop.is_closed():
            return
        future = asyncio.run_coroutine_threadsafe(self._run_hotkey_activation(binding), loop)
        self._activation_futures.add(future)
        future.add_done_callback(self._activation_done)

    async def _run_hotkey_activation(self, binding: HotkeyBinding) -> None:
        activation = self._hotkey_activation
        if activation is not None:
            await activation(binding)

    def _activation_done(self, future: concurrent.futures.Future[None]) -> None:
        self._activation_futures.discard(future)
        if future.cancelled():
            return
        try:
            future.result()
        except Exception:
            logger.exception("native global hotkey action failed")

    def _connect_hotkeys(self) -> sqlite3.Connection:
        if self._database_path is None:
            raise RuntimeError("hotkey persistence is not configured")
        connection = sqlite3.connect(self._database_path, timeout=10.0)
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _load_hotkeys(self) -> dict[str, HotkeyBinding]:
        if self._database_path is None:
            return {}
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect_hotkeys() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS global_hotkeys (
                    binding_id TEXT PRIMARY KEY,
                    accelerator TEXT NOT NULL UNIQUE,
                    project_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    field_name TEXT NOT NULL
                )
                """
            )
            rows = connection.execute(
                "SELECT binding_id, accelerator, project_id, node_id, field_name "
                "FROM global_hotkeys ORDER BY binding_id"
            ).fetchall()
        bindings: dict[str, HotkeyBinding] = {}
        for row in rows:
            if not all(isinstance(value, str) for value in row):
                raise TypeError("global_hotkeys row contains a non-text value")
            binding = HotkeyBinding(
                binding_id=cast(str, row[0]),
                accelerator=cast(str, row[1]),
                project_id=cast(str, row[2]),
                node_id=cast(str, row[3]),
                field=cast(str, row[4]),
            )
            bindings[binding.binding_id] = binding
        return bindings

    def _save_hotkey(self, binding: HotkeyBinding) -> None:
        if self._database_path is None:
            return
        try:
            with self._connect_hotkeys() as connection:
                connection.execute(
                    """
                    INSERT INTO global_hotkeys(binding_id, accelerator, project_id, node_id, field_name)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(binding_id) DO UPDATE SET
                        accelerator = excluded.accelerator,
                        project_id = excluded.project_id,
                        node_id = excluded.node_id,
                        field_name = excluded.field_name
                    """,
                    (
                        binding.binding_id,
                        binding.accelerator,
                        binding.project_id,
                        binding.node_id,
                        binding.field,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"hotkey accelerator is already registered: {binding.accelerator}") from exc

    def _delete_hotkey(self, binding_id: str) -> None:
        if self._database_path is None:
            return
        with self._connect_hotkeys() as connection:
            connection.execute("DELETE FROM global_hotkeys WHERE binding_id = ?", (binding_id,))

    @staticmethod
    def _unity_available() -> bool:
        try:
            import f8unitymods_setup
        except ImportError:
            return False
        return Path(f8unitymods_setup.__file__ or "").is_file()

    @staticmethod
    def _serial_available() -> bool:
        try:
            import serial
        except ImportError:
            return False
        return bool(serial.VERSION)

    @staticmethod
    def _action_write_hints(action: str) -> tuple[str, ...]:
        if "bepinex" in action:
            return ("<game>/BepInEx/**", "<game>/doorstop_config.ini", "<game>/winhttp.dll")
        if action in {"install_exporter", "install_profile"}:
            return ("<game>/BepInEx/plugins/F8SkeletonStreamer/**",)
        if action == "install_exporter_config":
            return ("<game>/BepInEx/config/**",)
        return ()


__all__ = [
    "ApplyUnityInstallRequest",
    "DEFAULT_SKELETON_UDP_PORT",
    "DetectModdingTargetRequest",
    "HotkeyBinding",
    "LocalCapability",
    "LocalIntegrationService",
    "PreviewUnityInstallRequest",
    "RegisterHotkeyRequest",
    "SerialPortInfo",
    "SkeletonUdpVerification",
    "UnityInstallPlan",
    "VerifySkeletonUdpRequest",
    "skeleton_graph_build_plan",
]
