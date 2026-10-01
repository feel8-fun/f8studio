from __future__ import annotations

import asyncio
import logging
import traceback
from dataclasses import replace
from typing import Any

from f8pysdk.bus import ServiceBus
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import ServiceNode

from .config import (
    ModelComplexity, SkeletonSource,
    DEFAULT_INFER_EVERY_N,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_MIN_TRACKING_CONFIDENCE,
    DEFAULT_MODEL_COMPLEXITY,
    DEFAULT_SKELETON_SOURCE,
    DEFAULT_VISIBILITY_THRESHOLD,
    PoseServiceConfig,
    coerce_float,
    coerce_int,
    coerce_model_complexity,
    coerce_skeleton_source,
    state_or_default,
)
from .payloads import (
    build_pose_detection_payload,
    build_pose_skeleton_payload,
    extract_pose_keypoints,
    extract_pose_world_keypoints,
    should_run_inference,
)
from .runtime import PoseRuntimeConfig, create_pose_runtime, tasks_model_spec_for_complexity
from .video_input import FrameContext, LatestVideoInput, frame_rgb_from_context

log = logging.getLogger(__name__)

_tasks_model_spec_for_complexity = tasks_model_spec_for_complexity


class MediaPipePoseServiceNode(ServiceNode):
    def __init__(self, *, node_id: str, node: Any, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=["video"],
            data_out_ports=["detections", "skeletons"],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._active = True
        self._config_loaded = False
        self._task: asyncio.Task[object] | None = None
        self._config = PoseServiceConfig()
        self._apply_config(self._config)

        self._video_input = LatestVideoInput()
        self._pose_runtime: Any | None = None
        self._zenoh_config_path: str | None = None
        self._zenoh_connect: tuple[str, ...] = ()
        self._zenoh_listen: tuple[str, ...] = ()
        self._zenoh_shm_pool_bytes = 256 * 1024 * 1024

        self._last_error_signature = ""
        self._last_error_repeats = 0
        self._last_infer_frame_id: int | None = None
        self._last_processed_frame_id: int | None = None

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        if isinstance(bus, ServiceBus):
            cfg = bus.config
            self._zenoh_config_path = cfg.zenoh_config_path
            self._zenoh_connect = cfg.zenoh_connect
            self._zenoh_listen = cfg.zenoh_listen
            self._zenoh_shm_pool_bytes = cfg.zenoh_shm_pool_bytes
        loop = asyncio.get_running_loop()
        loop.create_task(self._ensure_config_loaded(), name=f"f8mppose:init:{self.node_id}")
        self._task = loop.create_task(self._loop(), name=f"f8mppose:loop:{self.node_id}")

    async def close(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._close_video_input()
        self._close_pose_runtime()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self._active = bool(active)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del value
        del ts_ms
        name = str(field or "").strip()
        await self._ensure_config_loaded()

        if name == "inferEveryN":
            self._infer_every_n = coerce_int(
                await self.get_state_value("inferEveryN"),
                default=self._infer_every_n,
                minimum=1,
                maximum=10000,
            )
            self._config = replace(self._config, infer_every_n=self._infer_every_n)
            return

        if name == "modelComplexity":
            self._model_complexity = coerce_model_complexity(
                await self.get_state_value("modelComplexity"),
                default=self._model_complexity,
            )
            self._config = replace(self._config, model_complexity=self._model_complexity)
            await self._reset_pose_runtime()
            return

        if name == "minDetectionConfidence":
            self._min_detection_confidence = coerce_float(
                await self.get_state_value("minDetectionConfidence"),
                default=self._min_detection_confidence,
                minimum=0.0,
                maximum=1.0,
            )
            self._config = replace(
                self._config,
                min_detection_confidence=self._min_detection_confidence,
            )
            await self._reset_pose_runtime()
            return

        if name == "minTrackingConfidence":
            self._min_tracking_confidence = coerce_float(
                await self.get_state_value("minTrackingConfidence"),
                default=self._min_tracking_confidence,
                minimum=0.0,
                maximum=1.0,
            )
            self._config = replace(
                self._config,
                min_tracking_confidence=self._min_tracking_confidence,
            )
            await self._reset_pose_runtime()
            return

        if name == "visibilityThreshold":
            self._visibility_threshold = coerce_float(
                await self.get_state_value("visibilityThreshold"),
                default=self._visibility_threshold,
                minimum=0.0,
                maximum=1.0,
            )
            self._config = replace(self._config, visibility_threshold=self._visibility_threshold)
            return

        if name == "skeletonSource":
            self._skeleton_source = coerce_skeleton_source(
                await self.get_state_value("skeletonSource"),
                default=self._skeleton_source,
            )
            self._config = replace(self._config, skeleton_source=self._skeleton_source)
            return

    async def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return
        self._config = PoseServiceConfig(
            infer_every_n=coerce_int(
                state_or_default(
                    await self.get_state_value("inferEveryN"),
                    self._initial_state.get("inferEveryN"),
                    default=DEFAULT_INFER_EVERY_N,
                ),
                default=DEFAULT_INFER_EVERY_N,
                minimum=1,
                maximum=10000,
            ),
            model_complexity=coerce_model_complexity(
                state_or_default(
                    await self.get_state_value("modelComplexity"),
                    self._initial_state.get("modelComplexity"),
                    default=DEFAULT_MODEL_COMPLEXITY,
                ),
                default=DEFAULT_MODEL_COMPLEXITY,
            ),
            min_detection_confidence=coerce_float(
                state_or_default(
                    await self.get_state_value("minDetectionConfidence"),
                    self._initial_state.get("minDetectionConfidence"),
                    default=DEFAULT_MIN_DETECTION_CONFIDENCE,
                ),
                default=DEFAULT_MIN_DETECTION_CONFIDENCE,
                minimum=0.0,
                maximum=1.0,
            ),
            min_tracking_confidence=coerce_float(
                state_or_default(
                    await self.get_state_value("minTrackingConfidence"),
                    self._initial_state.get("minTrackingConfidence"),
                    default=DEFAULT_MIN_TRACKING_CONFIDENCE,
                ),
                default=DEFAULT_MIN_TRACKING_CONFIDENCE,
                minimum=0.0,
                maximum=1.0,
            ),
            visibility_threshold=coerce_float(
                state_or_default(
                    await self.get_state_value("visibilityThreshold"),
                    self._initial_state.get("visibilityThreshold"),
                    default=DEFAULT_VISIBILITY_THRESHOLD,
                ),
                default=DEFAULT_VISIBILITY_THRESHOLD,
                minimum=0.0,
                maximum=1.0,
            ),
            skeleton_source=coerce_skeleton_source(
                state_or_default(
                    await self.get_state_value("skeletonSource"),
                    self._initial_state.get("skeletonSource"),
                    default=DEFAULT_SKELETON_SOURCE,
                ),
                default=DEFAULT_SKELETON_SOURCE,
            ),
        )
        self._apply_config(self._config)
        self._config_loaded = True

    def _apply_config(self, config: PoseServiceConfig) -> None:
        self._infer_every_n = config.infer_every_n
        self._model_complexity: ModelComplexity = config.model_complexity
        self._min_detection_confidence = config.min_detection_confidence
        self._min_tracking_confidence = config.min_tracking_confidence
        self._visibility_threshold = config.visibility_threshold
        self._skeleton_source: SkeletonSource = config.skeleton_source

    async def _set_last_error(self, message: str) -> None:
        normalized = str(message or "")
        if normalized:
            await self.report_error(
                "MPPOSE_RUNTIME",
                normalized,
                severity="error",
                fingerprint=f"mppose:{normalized}",
            )
            return
        await self.clear_error()

    async def _record_exception(self, *, where: str, exc: Exception) -> None:
        signature = f"{where}:{type(exc).__name__}:{exc}"
        if signature == self._last_error_signature:
            self._last_error_repeats += 1
        else:
            self._last_error_signature = signature
            self._last_error_repeats = 1
        if self._last_error_repeats != 1 and self._last_error_repeats % 100 != 0:
            return
        await self._set_last_error(
            f"{where} failed with {type(exc).__name__}: {exc}\n"
            f"repeat={self._last_error_repeats}\n"
            f"traceback:\n{traceback.format_exc()}"
        )

    async def _reset_pose_runtime(self) -> None:
        self._close_pose_runtime()
        self._last_error_signature = ""
        self._last_error_repeats = 0
        await self._set_last_error("")

    def _close_pose_runtime(self) -> None:
        if self._pose_runtime is None:
            return
        try:
            self._pose_runtime.close()
        except Exception as exc:
            log.exception("pose runtime close failed", exc_info=exc)
        self._pose_runtime = None

    async def _ensure_pose_runtime(self) -> None:
        if self._pose_runtime is not None:
            return
        config = PoseRuntimeConfig(
            model_complexity=self._model_complexity,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        self._pose_runtime = create_pose_runtime(config)

    async def _maybe_reopen_video_input(self) -> None:
        stream_key = self._resolve_video_stream_key()
        if self._video_input.is_open_for(stream_key=stream_key):
            return
        self._close_video_input()

    def _resolve_video_stream_key(self) -> str:
        return str(self.input_zenoh_key("video") or "").strip()

    def _close_video_input(self) -> None:
        try:
            self._video_input.close()
        except Exception as exc:
            log.exception("video input close failed", exc_info=exc)

    def _accept_frame_for_processing(self, frame_id: int) -> bool:
        if self._last_processed_frame_id is not None and frame_id == int(self._last_processed_frame_id):
            return False

        self._last_processed_frame_id = frame_id
        return True

    async def _process_frame(self, frame: FrameContext, *, np_module: Any) -> None:
        frame_rgb = frame_rgb_from_context(frame, np_module=np_module)

        assert self._pose_runtime is not None
        result = self._pose_runtime.process(frame_rgb, timestamp_ms=frame.ts_ms)
        keypoints = extract_pose_keypoints(
            result,
            width=frame.width,
            height=frame.height,
            visibility_threshold=self._visibility_threshold,
        )
        world_keypoints = extract_pose_world_keypoints(
            result,
            visibility_threshold=self._visibility_threshold,
        )
        payload_out = build_pose_detection_payload(
            frame_id=frame.frame_id,
            ts_ms=frame.ts_ms,
            width=frame.width,
            height=frame.height,
            keypoints=keypoints,
        )
        skeleton_payload = build_pose_skeleton_payload(
            frame_id=frame.frame_id,
            ts_ms=frame.ts_ms,
            keypoints=keypoints,
            world_keypoints=world_keypoints,
            skeleton_source=self._skeleton_source,
        )
        await self.emit("detections", payload_out, ts_ms=frame.ts_ms)
        await self.emit("skeletons", [skeleton_payload], ts_ms=frame.ts_ms)
        self._last_infer_frame_id = frame.frame_id

    async def _loop(self) -> None:
        import numpy as np  # type: ignore[import-not-found]

        while True:
            try:
                await asyncio.sleep(0)
                if not self._active:
                    await asyncio.sleep(0.05)
                    continue

                await self._ensure_config_loaded()
                await self._ensure_pose_runtime()

                stream_key = self._resolve_video_stream_key()
                if not stream_key:
                    await asyncio.sleep(0.05)
                    continue

                if not self._video_input.is_open_for(stream_key=stream_key):
                    self._video_input.open(
                        stream_key=stream_key,
                        config_path=self._zenoh_config_path,
                        connect=self._zenoh_connect,
                        listen=self._zenoh_listen,
                        shm_pool_bytes=self._zenoh_shm_pool_bytes,
                    )

                frame = self._video_input.read_frame()
                if frame is None:
                    continue

                if not self._accept_frame_for_processing(frame.frame_id):
                    continue

                if not should_run_inference(self._last_infer_frame_id, frame.frame_id, self._infer_every_n):
                    continue

                await self._process_frame(frame, np_module=np)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._record_exception(where="loop", exc=exc)
                await asyncio.sleep(0.1)
