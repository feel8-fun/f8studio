from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from f8pysdk.codec import coerce_float, coerce_int, coerce_str, parse_float
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import ServiceNode
from f8pysdk.time_utils import now_ms
from f8pysdk.video_transport import (
    VIDEO_FORMAT_FLOW2_F16,
    VIDEO_FORMAT_SCALAR1_F32,
)

from .video_frame_source import LatestVideoFrameSource, VideoFrameSourceConfig

SortDirection = Literal["asc", "desc"]
ScoreAggregation = Literal["mean", "max", "sum", "median"]
ScoreSourceUnavailableReason = Literal["not_ready", "invalid"]

logger = logging.getLogger(__name__)

_CLS_WEIGHTS_REGEX_PREFIX = "re:"


class ScoreSourceUnavailableError(RuntimeError):
    """Raised when the selected score-map source is missing, unreadable, or unsupported."""

    def __init__(self, message: str, *, reason: ScoreSourceUnavailableReason) -> None:
        super().__init__(message)
        self.reason = reason


class ScoreFrameHeader(Protocol):
    width: int
    height: int
    pitch: int
    fmt: int


def _coerce_sort_direction(value: Any, *, default: SortDirection = "desc") -> SortDirection:
    text = coerce_str(value, default=default).lower()
    if text == "asc":
        return "asc"
    if text == "desc":
        return "desc"
    return default


def _coerce_score_aggregation(value: Any, *, default: ScoreAggregation = "mean") -> ScoreAggregation:
    text = coerce_str(value, default=default).lower()
    if text == "max":
        return "max"
    if text == "sum":
        return "sum"
    if text == "median":
        return "median"
    return "mean"


def _coerce_temperature(value: Any, *, default: float = 0.0) -> float:
    return coerce_float(
        value,
        default=default,
        minimum=0.0,
        allow_bool=False,
        finite_only=True,
    )


def _payload_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_cls_weights_json(text: str) -> tuple[dict[str, float], list[tuple[re.Pattern[str], float]]]:
    """
    Parse clsWeights JSON string into:
    - exact mapping: {"person": 2.0, ...}
    - regex rules (in order): [(re.compile("^dog_.*$"), 1.3), ...]

    Rules:
    - keys with "re:" prefix are treated as regex patterns matched via fullmatch().
    - keys without prefix are exact cls matches.
    - final multiplier is the product of the exact match (if any) and every regex rule that matches.
    """
    normalized = str(text or "").strip()
    if not normalized:
        normalized = "{}"

    try:
        obj = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ValueError(f"clsWeights must be valid JSON object: {exc}") from exc

    if not isinstance(obj, dict):
        raise ValueError("clsWeights must be a JSON object mapping string -> number")

    exact: dict[str, float] = {}
    regex: list[tuple[re.Pattern[str], float]] = []

    for raw_key, raw_weight in obj.items():
        key = str(raw_key or "")
        if not key:
            raise ValueError("clsWeights keys must be non-empty strings")
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise ValueError(f"clsWeights[{key!r}] must be a number")
        weight = float(raw_weight)
        if not math.isfinite(weight):
            raise ValueError(f"clsWeights[{key!r}] must be a finite number")

        if key.startswith(_CLS_WEIGHTS_REGEX_PREFIX):
            pattern_text = key[len(_CLS_WEIGHTS_REGEX_PREFIX) :]
            if not pattern_text:
                raise ValueError("clsWeights regex keys must include a non-empty pattern after 're:'")
            try:
                pattern = re.compile(pattern_text)
            except re.error as exc:
                raise ValueError(f"invalid clsWeights regex {pattern_text!r}: {exc}") from exc
            regex.append((pattern, weight))
        else:
            exact[key] = weight

    return exact, regex


def _combined_cls_weight(
    cls_text: str,
    *,
    cls_weights_exact: dict[str, float] | None,
    cls_weights_regex: list[tuple[re.Pattern[str], float]] | None,
) -> float:
    weight = 1.0
    if cls_weights_exact is not None and cls_text in cls_weights_exact:
        weight *= float(cls_weights_exact[cls_text])
    if cls_weights_regex is not None:
        for pattern, rule_weight in cls_weights_regex:
            if pattern.fullmatch(cls_text):
                weight *= float(rule_weight)
    return float(weight)


def decode_score_map_from_frame(*, header: ScoreFrameHeader, payload: memoryview) -> np.ndarray:
    width = int(header.width)
    height = int(header.height)
    pitch = int(header.pitch)
    if width <= 0 or height <= 0 or pitch <= 0:
        raise ValueError("invalid score frame dimensions")

    frame_bytes = int(pitch) * int(height)
    if len(payload) < frame_bytes:
        raise ValueError("score frame payload too small")

    if int(header.fmt) == VIDEO_FORMAT_SCALAR1_F32:
        if pitch < width * 4 or (pitch % 4) != 0:
            raise ValueError("invalid scalar1_f32 pitch")
        row_floats = pitch // 4
        scores = np.frombuffer(payload, dtype=np.float32, count=row_floats * height).reshape((height, row_floats))
        return np.ascontiguousarray(scores[:, :width], dtype=np.float32)

    if int(header.fmt) == VIDEO_FORMAT_FLOW2_F16:
        if pitch < width * 4 or (pitch % 2) != 0:
            raise ValueError("invalid flow2_f16 pitch")
        row_halfs = pitch // 2
        flow_halfs = np.frombuffer(payload, dtype=np.float16, count=row_halfs * height).reshape((height, row_halfs))
        packed = np.ascontiguousarray(flow_halfs[:, : width * 2], dtype=np.float16)
        flow = packed.astype(np.float32).reshape((height, width, 2))
        magnitude = np.sqrt((flow[:, :, 0] * flow[:, :, 0]) + (flow[:, :, 1] * flow[:, :, 1]))
        return np.ascontiguousarray(magnitude, dtype=np.float32)

    raise ValueError(f"unsupported score frame format: {int(header.fmt)}")


def rescale_bbox_to_score_map(
    bbox: list[Any] | tuple[Any, ...],
    *,
    detections_width: int,
    detections_height: int,
    score_width: int,
    score_height: int,
) -> tuple[int, int, int, int] | None:
    if len(bbox) < 4:
        return None
    if detections_width <= 0 or detections_height <= 0 or score_width <= 0 or score_height <= 0:
        return None

    x1 = coerce_int(bbox[0], default=0, allow_bool=False)
    y1 = coerce_int(bbox[1], default=0, allow_bool=False)
    x2 = coerce_int(bbox[2], default=0, allow_bool=False)
    y2 = coerce_int(bbox[3], default=0, allow_bool=False)
    if x2 <= x1 or y2 <= y1:
        return None

    scale_x = float(score_width) / float(detections_width)
    scale_y = float(score_height) / float(detections_height)

    scaled_x1 = int(np.floor(float(x1) * scale_x))
    scaled_y1 = int(np.floor(float(y1) * scale_y))
    scaled_x2 = int(np.ceil(float(x2) * scale_x))
    scaled_y2 = int(np.ceil(float(y2) * scale_y))

    clamped_x1 = max(0, min(score_width, scaled_x1))
    clamped_y1 = max(0, min(score_height, scaled_y1))
    clamped_x2 = max(0, min(score_width, scaled_x2))
    clamped_y2 = max(0, min(score_height, scaled_y2))
    if clamped_x2 <= clamped_x1 or clamped_y2 <= clamped_y1:
        return None
    return (clamped_x1, clamped_y1, clamped_x2, clamped_y2)


def aggregate_roi_score(roi: np.ndarray, aggregation: ScoreAggregation) -> float | None:
    if roi.size <= 0:
        return None

    roi_float = np.asarray(roi, dtype=np.float32)
    if aggregation == "max":
        return float(np.max(roi_float))
    if aggregation == "sum":
        return float(np.sum(roi_float, dtype=np.float32))
    if aggregation == "median":
        return float(np.median(roi_float))
    return float(np.mean(roi_float, dtype=np.float32))


@dataclass(frozen=True)
class RankedDetection:
    index: int
    detection: dict[str, Any]
    metric_score: float | None
    rank_score: float | None


def _sort_valid_detections(
    ranked: list[RankedDetection],
    *,
    sort_direction: SortDirection,
    temperature: float,
    random_generator: np.random.Generator | None,
) -> None:
    reverse = sort_direction == "desc"
    if temperature <= 0.0 or len(ranked) < 2:
        ranked.sort(key=lambda item: float(item.rank_score), reverse=reverse)
        return

    rank_scores = np.asarray([float(item.rank_score) for item in ranked], dtype=np.float64)
    if not np.all(np.isfinite(rank_scores)):
        ranked.sort(key=lambda item: float(item.rank_score), reverse=reverse)
        return

    lowest_score = float(np.min(rank_scores))
    score_span = float(np.max(rank_scores)) - lowest_score
    if score_span > 0.0:
        utilities = (rank_scores - lowest_score) / score_span
        if sort_direction == "asc":
            utilities = 1.0 - utilities
    else:
        utilities = np.zeros(len(ranked), dtype=np.float64)

    rng = random_generator if random_generator is not None else np.random.default_rng()
    sampling_keys = utilities + (temperature * rng.gumbel(size=len(ranked)))
    randomized = list(zip(ranked, sampling_keys, strict=True))
    randomized.sort(key=lambda item: float(item[1]), reverse=True)
    ranked[:] = [item[0] for item in randomized]


def sort_detection_payload(
    detections_payload: dict[str, Any],
    *,
    score_map: np.ndarray,
    sort_direction: SortDirection,
    score_aggregation: ScoreAggregation,
    temperature: float = 0.0,
    random_generator: np.random.Generator | None = None,
    cls_weights_exact: dict[str, float] | None = None,
    cls_weights_regex: list[tuple[re.Pattern[str], float]] | None = None,
) -> dict[str, Any] | None:
    detections_value = detections_payload.get("detections")
    if not isinstance(detections_value, list) or not detections_value:
        return None

    detections_width = _payload_int(detections_payload, "width")
    detections_height = _payload_int(detections_payload, "height")
    if detections_width is None or detections_height is None:
        return None

    score_height = int(score_map.shape[0])
    score_width = int(score_map.shape[1])
    ranked: list[RankedDetection] = []
    for index, raw_detection in enumerate(detections_value):
        if not isinstance(raw_detection, dict):
            ranked.append(RankedDetection(index=index, detection={}, metric_score=None, rank_score=None))
            continue
        bbox_value = raw_detection.get("bbox")
        if not isinstance(bbox_value, list) and not isinstance(bbox_value, tuple):
            ranked.append(RankedDetection(index=index, detection=dict(raw_detection), metric_score=None, rank_score=None))
            continue
        scaled_bbox = rescale_bbox_to_score_map(
            bbox_value,
            detections_width=detections_width,
            detections_height=detections_height,
            score_width=score_width,
            score_height=score_height,
        )
        if scaled_bbox is None:
            ranked.append(RankedDetection(index=index, detection=dict(raw_detection), metric_score=None, rank_score=None))
            continue
        x1, y1, x2, y2 = scaled_bbox
        roi = score_map[y1:y2, x1:x2]
        metric_score = aggregate_roi_score(roi, score_aggregation)
        if metric_score is None:
            ranked.append(RankedDetection(index=index, detection=dict(raw_detection), metric_score=None, rank_score=None))
            continue

        cls_name = raw_detection.get("cls")
        cls_text = cls_name if isinstance(cls_name, str) else str(cls_name or "")
        weight = _combined_cls_weight(
            cls_text,
            cls_weights_exact=cls_weights_exact,
            cls_weights_regex=cls_weights_regex,
        )
        rank_score = float(metric_score) * float(weight)
        ranked.append(
            RankedDetection(
                index=index,
                detection=dict(raw_detection),
                metric_score=float(metric_score),
                rank_score=rank_score,
            )
        )

    valid_ranked: list[RankedDetection] = []
    invalid_ranked: list[RankedDetection] = []
    for item in ranked:
        if item.rank_score is None:
            invalid_ranked.append(item)
        else:
            valid_ranked.append(item)

    _sort_valid_detections(
        valid_ranked,
        sort_direction=sort_direction,
        temperature=_coerce_temperature(temperature),
        random_generator=random_generator,
    )
    ordered = valid_ranked + invalid_ranked
    if not ordered:
        return None

    output_payload = dict(detections_payload)
    output_payload["detections"] = [item.detection for item in ordered]
    return output_payload


class DetectionSorterServiceNode(ServiceNode):
    def __init__(self, *, node_id: str, node: Any, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=["detections", "score"],
            data_out_ports=["detections"],
            state_fields=[state.name for state in list(node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._active = True
        self._config_loaded = False
        self._sort_direction: SortDirection = "desc"
        self._score_aggregation: ScoreAggregation = "mean"
        self._temperature = 0.0
        self._random_generator = np.random.default_rng()
        self._cls_weights_exact: dict[str, float] = {}
        self._cls_weights_regex: list[tuple[re.Pattern[str], float]] = []
        self._last_error = ""
        self._latest_detections: dict[str, Any] | None = None
        self._score_source_config = VideoFrameSourceConfig.from_bus(None)
        self._score_source: LatestVideoFrameSource | None = None
        self._on_data_lock = asyncio.Lock()

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        self._score_source_config = VideoFrameSourceConfig.from_bus(bus)
        self._score_source = LatestVideoFrameSource(config=self._score_source_config)

    async def close(self) -> None:
        self._close_score_reader()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self._active = bool(active)

    def _read_initial_or_cached_state(self, field: str, default: Any) -> Any:
        missing = object()
        cached_value = self.get_state_cached(field, missing)
        if cached_value is not missing:
            return cached_value
        if field in self._initial_state:
            return self._initial_state[field]
        return default

    async def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return
        self._sort_direction = _coerce_sort_direction(self._read_initial_or_cached_state("sortDirection", "desc"))
        self._score_aggregation = _coerce_score_aggregation(self._read_initial_or_cached_state("scoreAggregation", "mean"))
        self._temperature = _coerce_temperature(self._read_initial_or_cached_state("temperature", 0.0))
        cls_weights_text = coerce_str(self._read_initial_or_cached_state("clsWeights", "{}"), default="{}")
        self._set_cls_weights(cls_weights_text)
        self._config_loaded = True

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "sortDirection":
            direction = coerce_str(value).lower()
            if direction not in ("asc", "desc"):
                raise ValueError("invalid sortDirection (expected asc or desc)")
            return direction
        if name == "scoreAggregation":
            aggregation = coerce_str(value).lower()
            if aggregation not in ("mean", "max", "sum", "median"):
                raise ValueError("invalid scoreAggregation (expected mean, max, sum, or median)")
            return aggregation
        if name == "temperature":
            temperature = parse_float(value, allow_bool=False, finite_only=True)
            if temperature is None or temperature < 0.0:
                raise ValueError("invalid temperature (expected a finite number greater than or equal to 0)")
            return temperature
        if name == "clsWeights":
            text = coerce_str(value, default="{}")
            _ = _parse_cls_weights_json(text)
            return text
        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        await self._ensure_config_loaded()
        name = str(field or "").strip()
        if name == "sortDirection":
            self._sort_direction = _coerce_sort_direction(value, default=self._sort_direction)
            return
        if name == "scoreAggregation":
            self._score_aggregation = _coerce_score_aggregation(value, default=self._score_aggregation)
            return
        if name == "temperature":
            self._temperature = _coerce_temperature(value, default=self._temperature)
            return
        if name == "clsWeights":
            text = coerce_str(value, default="{}")
            self._set_cls_weights(text)
            return

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        if port != "detections":
            return
        if not self._active:
            return
        started_at = time.perf_counter()
        await self._ensure_config_loaded()
        if not isinstance(value, dict):
            await self._set_last_error("detections payload must be an object")
            return
        async with self._on_data_lock:
            incoming_payload = dict(value)
            self._latest_detections = incoming_payload
            try:
                output_payload = await asyncio.to_thread(self._sort_latest_detections)
            except ScoreSourceUnavailableError as exc:
                if exc.reason == "not_ready":
                    await self._set_last_error("")
                else:
                    await self._set_last_error(self._format_score_source_unavailable_error(exc))
                await self._emit_output_with_timing(incoming_payload, started_at=started_at)
                return
            except Exception as exc:
                logger.exception("detection sorter failed node=%s", self.node_id)
                await self._set_last_error(f"{type(exc).__name__}: {exc}")
                await self._emit_output_with_timing(incoming_payload, started_at=started_at)
                return
            if output_payload is None:
                await self._set_last_error("")
                await self._emit_output_with_timing(incoming_payload, started_at=started_at)
                return
            await self._set_last_error("")
            await self._emit_output_with_timing(output_payload, started_at=started_at)

    async def _emit_output_with_timing(self, payload: dict[str, Any], *, started_at: float) -> None:
        source_ts_ms = _payload_int(payload, "tsMs")
        await self.emit("detections", payload, ts_ms=source_ts_ms)
        completed_ts_ms = int(now_ms())
        process_ms = max(0.0, (time.perf_counter() - float(started_at)) * 1000.0)
        latency_ms = 0.0
        if source_ts_ms is not None and int(source_ts_ms) > 0:
            latency_ms = max(0.0, float(completed_ts_ms - int(source_ts_ms)))
        self.record_monitor_timing(
            port="detections",
            process_ms=process_ms,
            latency_ms=latency_ms,
            ts_ms=completed_ts_ms,
        )

    def _close_score_reader(self) -> None:
        source = self._score_source
        self._score_source = None
        if source is not None:
            source.close()

    def _reset_score_source(self) -> None:
        source = self._score_source
        if source is not None:
            source.reset()

    def _ensure_score_source(self) -> LatestVideoFrameSource:
        source = self._score_source
        if source is not None:
            return source
        source = LatestVideoFrameSource(config=self._score_source_config)
        self._score_source = source
        return source

    def _read_latest_score_frame(self) -> tuple[ScoreFrameHeader, memoryview]:
        score_key = str(self.input_zenoh_key("score") or "").strip()
        if not score_key:
            raise ScoreSourceUnavailableError("score data input is missing", reason="not_ready")
        source = self._ensure_score_source()
        try:
            frame = source.read_latest(stream_key=score_key, timeout_ms=0, dedupe=False)
        except FileNotFoundError as exc:
            raise ScoreSourceUnavailableError(f"open pending: {type(exc).__name__}: {exc}", reason="not_ready") from exc
        except ValueError as exc:
            message = str(exc).strip()
            reason: ScoreSourceUnavailableReason = "invalid" if "invalid" in message.lower() else "not_ready"
            raise ScoreSourceUnavailableError(
                f"score source unavailable: {type(exc).__name__}: {exc}",
                reason=reason,
            ) from exc
        except (RuntimeError, OSError) as exc:
            raise ScoreSourceUnavailableError(
                f"score source unavailable: {type(exc).__name__}: {exc}",
                reason="not_ready",
            ) from exc
        if frame is None:
            raise ScoreSourceUnavailableError("score source has no readable frame", reason="not_ready")
        return frame, frame.payload

    def _sort_latest_detections(self) -> dict[str, Any] | None:
        if self._latest_detections is None:
            return None
        detections_value = self._latest_detections.get("detections")
        if not isinstance(detections_value, list) or not detections_value:
            return None
        header, payload = self._read_latest_score_frame()
        try:
            try:
                score_map = decode_score_map_from_frame(header=header, payload=payload)
            except ValueError as exc:
                raise ScoreSourceUnavailableError(str(exc), reason="invalid") from exc
            return sort_detection_payload(
                self._latest_detections,
                score_map=score_map,
                sort_direction=self._sort_direction,
                score_aggregation=self._score_aggregation,
                temperature=self._temperature,
                random_generator=self._random_generator,
                cls_weights_exact=self._cls_weights_exact,
                cls_weights_regex=self._cls_weights_regex,
            )
        finally:
            payload.release()

    def _set_cls_weights(self, text: str) -> None:
        exact, regex = _parse_cls_weights_json(text)
        self._cls_weights_exact = exact
        self._cls_weights_regex = regex

    @staticmethod
    def _format_score_source_unavailable_error(exc: ScoreSourceUnavailableError) -> str:
        details = str(exc).strip()
        if not details:
            return "score source unavailable"
        if len(details) > 200:
            details = details[:200] + "..."
        return f"score source unavailable: {details}"

    async def _set_last_error(self, message: str) -> None:
        normalized = str(message or "")
        if normalized == self._last_error:
            return
        self._last_error = normalized
        if self._last_error:
            await self.report_error(
                "DL_DETECTION_SORTER_RUNTIME",
                self._last_error,
                severity="error",
                fingerprint=f"dl-detection-sorter:{self._last_error}",
            )
            return
        await self.clear_error()
