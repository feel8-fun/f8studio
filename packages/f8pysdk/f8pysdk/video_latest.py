"""Shared script-facing latest-video packets and subscription lifecycle.

NumPy is optional; raw packets remain available without the decoding extra.
Callers supply their own lifecycle gate through ``read_enabled``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from f8pysdk.video_transport import VIDEO_FORMAT_BGRA32, VIDEO_FORMAT_FLOW2_F16
from f8pysdk.video_transport import (
    LatestVideoFrame,
    ZenohLatestVideoFrameTransport,
)

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

_VIDEO_LATEST_READER_CLOSE_ERRORS = (Exception,)
_VIDEO_LATEST_SUBSCRIPTION_ERRORS = (Exception,)


class VideoLatestReader(Protocol):
    def wait_latest(self, timeout_ms: int) -> LatestVideoFrame | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class VideoLatestConfig:
    config_path: str | None = None
    connect: tuple[str, ...] = ()
    listen: tuple[str, ...] = ()
    shm_pool_bytes: int = 256 * 1024 * 1024


@dataclass
class VideoLatestSubscription:
    key: str
    stream_key: str
    decode_mode: str
    reader: VideoLatestReader | None = None
    task: asyncio.Task[object] | None = None
    latest_packet: dict[str, Any] | None = None
    last_frame_id: int = 0
    last_error_sig: str | None = None
    last_error_ts_ms: int = 0
    error_count: int = 0


def normalize_video_decode_mode(decode: Any) -> str:
    mode = str(decode or "auto").strip().lower()
    if mode in ("none", "auto"):
        return mode
    return "auto"


def _source_metadata(sub: VideoLatestSubscription) -> dict[str, Any]:
    return {"key": sub.key, "transport": "zenoh", "streamKey": sub.stream_key}


def _status_metadata(sub: VideoLatestSubscription) -> dict[str, Any]:
    metadata = _source_metadata(sub)
    metadata["decodeMode"] = sub.decode_mode
    metadata["hasPacket"] = sub.latest_packet is not None
    metadata["lastFrameId"] = int(sub.last_frame_id)
    metadata["errorCount"] = int(sub.error_count)
    return metadata


class VideoLatestPacketCodec:
    @staticmethod
    def header_to_dict(frame: LatestVideoFrame) -> dict[str, int]:
        return {
            "frameId": int(frame.frame_id),
            "tsMs": int(frame.ts_ms),
            "width": int(frame.width),
            "height": int(frame.height),
            "pitch": int(frame.pitch),
            "fmt": int(frame.fmt),
            "notifySeq": 0,
        }

    @staticmethod
    def compact_rows(raw: bytes, *, width: int, height: int, pitch: int, row_bytes: int) -> bytes | None:
        if width <= 0 or height <= 0 or pitch < row_bytes or row_bytes <= 0:
            return None
        if len(raw) < (pitch * height):
            return None
        if pitch == row_bytes:
            return raw
        compact = bytearray(row_bytes * height)
        for y in range(height):
            src_off = y * pitch
            dst_off = y * row_bytes
            compact[dst_off : dst_off + row_bytes] = raw[src_off : src_off + row_bytes]
        return bytes(compact)

    def decode_payload(self, *, header: dict[str, int], raw: bytes, decode_mode: str) -> dict[str, Any] | None:
        if decode_mode != "auto":
            return None
        width = int(header.get("width") or 0)
        height = int(header.get("height") or 0)
        pitch = int(header.get("pitch") or 0)
        fmt = int(header.get("fmt") or 0)
        if width <= 0 or height <= 0 or pitch <= 0:
            return None

        if fmt == VIDEO_FORMAT_BGRA32:
            row_bytes = width * 4
            compact = self.compact_rows(raw, width=width, height=height, pitch=pitch, row_bytes=row_bytes)
            if compact is None:
                return {"kind": "bgra32", "shape": [height, width, 4], "data": None}
            data = None
            if np is not None:
                arr = np.frombuffer(compact, dtype=np.uint8)
                if int(arr.size) == (height * width * 4):
                    data = arr.reshape(height, width, 4)
            return {"kind": "bgra32", "shape": [height, width, 4], "data": data}

        if fmt == VIDEO_FORMAT_FLOW2_F16:
            row_bytes = width * 4
            compact = self.compact_rows(raw, width=width, height=height, pitch=pitch, row_bytes=row_bytes)
            if compact is None:
                return {"kind": "flow2_f16", "shape": [height, width, 2], "data": None}
            data = None
            if np is not None:
                arr = np.frombuffer(compact, dtype="<f2")
                if int(arr.size) == (height * width * 2):
                    data = arr.reshape(height, width, 2)
            return {"kind": "flow2_f16", "shape": [height, width, 2], "data": data}

        return None

    @staticmethod
    def copy_for_script(packet: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(packet, dict):
            return None
        header_src = packet.get("header")
        meta_src = packet.get("meta")
        decoded_src = packet.get("decoded")
        out: dict[str, Any] = {
            "header": dict(header_src) if isinstance(header_src, dict) else {},
            "meta": dict(meta_src) if isinstance(meta_src, dict) else {},
            "raw": packet.get("raw"),
            "decoded": None,
        }
        if isinstance(decoded_src, dict):
            decoded_out: dict[str, Any] = {}
            if "kind" in decoded_src:
                decoded_out["kind"] = decoded_src.get("kind")
            if "shape" in decoded_src:
                decoded_out["shape"] = list(decoded_src.get("shape") or [])
            if "data" in decoded_src:
                decoded_out["data"] = decoded_src.get("data")
            out["decoded"] = decoded_out
        return out


class VideoLatestSubscriptions:
    def __init__(
        self,
        *,
        node_id: str,
        log_context: str,
        read_enabled: Callable[[], bool] | None = None,
        task_prefix: str | None = None,
        config: VideoLatestConfig | None = None,
    ) -> None:
        self._node_id = str(node_id)
        self._log_context = str(log_context)
        self._task_prefix = str(task_prefix or log_context)
        self._read_enabled = read_enabled
        self._config = config if config is not None else VideoLatestConfig()
        self._subscriptions: dict[str, VideoLatestSubscription] = {}
        self._packet_codec = VideoLatestPacketCodec()

    def configure(self, config: VideoLatestConfig) -> None:
        self._config = config

    def subscribe(self, key: str, *, stream_key: str, decode: Any = "auto") -> None:
        key_name = str(key or "").strip()
        stream_key_text = str(stream_key or "").strip()
        if not key_name:
            return
        if not stream_key_text:
            return

        self.unsubscribe_sync(key_name)
        sub = VideoLatestSubscription(
            key=key_name,
            stream_key=stream_key_text,
            decode_mode=normalize_video_decode_mode(decode),
        )
        self._subscriptions[key_name] = sub

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            sub.task = None
            logger.error(
                "[%s:%s] subscribe_video_latest without running loop",
                self._node_id,
                self._log_context,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            return

        sub.task = loop.create_task(
            self._run_subscription(key_name),
            name=f"{self._task_prefix}:video_sub:{self._node_id}:{key_name}",
        )

    def get_packet(self, key: str) -> dict[str, Any] | None:
        key_name = str(key or "").strip()
        if not key_name:
            return None
        sub = self._subscriptions.get(key_name)
        if sub is None:
            return None
        return self._packet_codec.copy_for_script(sub.latest_packet)

    def unsubscribe_sync(self, key: str) -> bool:
        key_name = str(key or "").strip()
        if not key_name:
            return False
        sub = self._subscriptions.pop(key_name, None)
        if sub is None:
            return False
        task = sub.task
        sub.task = None
        if task is not None and not task.done():
            task.cancel()
        self._close_reader(sub)
        return True

    def list_status(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for key_name in sorted(self._subscriptions.keys()):
            sub = self._subscriptions.get(key_name)
            if sub is None:
                continue
            items.append(_status_metadata(sub))
        return items

    def shutdown_sync(self) -> None:
        keys = list(self._subscriptions.keys())
        for key in keys:
            self.unsubscribe_sync(key)

    async def shutdown_async(self) -> None:
        keys = list(self._subscriptions.keys())
        tasks: list[asyncio.Task[object]] = []
        for key in keys:
            sub = self._subscriptions.pop(key, None)
            if sub is None:
                continue
            task = sub.task
            sub.task = None
            if task is not None and not task.done():
                task.cancel()
                tasks.append(task)
            self._close_reader(sub)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000.0)

    def _log_sub_error(self, sub: VideoLatestSubscription, stage: str, exc: BaseException) -> None:
        sub.error_count += 1
        now_ms = self._now_ms()
        sig = f"{stage}:{type(exc).__name__}:{exc}"
        if sub.last_error_sig == sig and (now_ms - int(sub.last_error_ts_ms)) < 2000:
            return
        sub.last_error_sig = sig
        sub.last_error_ts_ms = now_ms
        logger.error(
            "[%s:%s] video latest subscribe failed key=%s stream_key=%s stage=%s",
            self._node_id,
            self._log_context,
            sub.key,
            sub.stream_key,
            stage,
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    def _close_reader(self, sub: VideoLatestSubscription) -> None:
        reader = sub.reader
        sub.reader = None
        if reader is None:
            return
        try:
            reader.close()
        except _VIDEO_LATEST_READER_CLOSE_ERRORS as exc:
            logger.error(
                "[%s:%s] video reader close failed key=%s",
                self._node_id,
                self._log_context,
                sub.key,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    def _open_reader(self, sub: VideoLatestSubscription) -> VideoLatestReader:
        return ZenohLatestVideoFrameTransport.open_subscriber(
            sub.stream_key,
            config_path=self._config.config_path,
            connect=self._config.connect,
            listen=self._config.listen,
            shm_pool_bytes=self._config.shm_pool_bytes,
        )

    def _update_latest_packet(self, sub: VideoLatestSubscription, frame: LatestVideoFrame) -> bool:
        frame_id = int(frame.frame_id)
        if frame_id <= 0:
            return False
        if frame_id == int(sub.last_frame_id) and sub.latest_packet is not None:
            return False

        frame_bytes = int(frame.frame_bytes)
        if frame_bytes <= 0 or frame_bytes > len(frame.payload):
            return False

        raw = bytes(frame.payload[:frame_bytes])
        header_dict = self._packet_codec.header_to_dict(frame)
        decoded = self._packet_codec.decode_payload(header=header_dict, raw=raw, decode_mode=sub.decode_mode)
        sub.latest_packet = {
            "header": header_dict,
            "raw": raw,
            "decoded": decoded,
            "meta": _source_metadata(sub)
            | {
                "decodeMode": sub.decode_mode,
                "lastUpdateMs": self._now_ms(),
            },
        }
        sub.last_frame_id = frame_id
        return True

    async def _run_subscription(self, key: str) -> None:
        key_name = str(key or "").strip()
        while True:
            sub = self._subscriptions.get(key_name)
            if sub is None:
                return

            if self._read_enabled is not None and not self._read_enabled():
                await asyncio.sleep(0.05)
                continue

            if sub.reader is None:
                try:
                    sub.reader = self._open_reader(sub)
                except _VIDEO_LATEST_SUBSCRIPTION_ERRORS as exc:
                    self._log_sub_error(sub, "open", exc)
                    await asyncio.sleep(0.2)
                    continue

            assert sub.reader is not None
            try:
                frame = sub.reader.wait_latest(20)
                if frame is None:
                    await asyncio.sleep(0)
                    continue
                try:
                    updated = self._update_latest_packet(sub, frame)
                finally:
                    frame.release()
                if not updated:
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                raise
            except _VIDEO_LATEST_SUBSCRIPTION_ERRORS as exc:
                self._log_sub_error(sub, "read", exc)
                self._close_reader(sub)
                await asyncio.sleep(0.2)


__all__ = [
    "VideoLatestConfig",
    "VideoLatestReader",
    "VideoLatestPacketCodec",
    "VideoLatestSubscriptions",
    "VideoLatestSubscription",
    "normalize_video_decode_mode",
]
