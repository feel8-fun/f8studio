from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlsplit

import httpx
import msgspec
from PIL import Image

from f8pysdk.bus import ServiceBus
from f8pysdk.codec import unwrap_json_value
from f8pysdk.decision import (
    ChoiceAnswer, DecisionRequest, DecisionResult, Question,
    ScoreAnswer, validate_questions, validate_result,
)
from f8pysdk.nodes import OperatorNode
from f8pysdk.generated import F8AnyTypeSchema
from f8pysdk.registry import Registry
from f8pysdk.specs import (
    F8DataPortSpec, F8JsonValue, F8OperatorSchemaVersion, F8OperatorSpec,
    F8RuntimeNode, F8StateAccess, F8StateSpec, any_schema, boolean_schema,
    exec_port_specs, integer_schema, number_schema, string_schema,
    video_frame_port,
)
from f8pysdk.video_transport import VIDEO_FORMAT_BGRA32, ZenohLatestVideoFrameTransport

from ..constants import SERVICE_CLASS

logger = logging.getLogger(__name__)
OPERATOR_CLASS = "f8.decision"
DEFAULT_QUESTIONS: dict[str, F8JsonValue] = {
    "decision": {"type": "choice", "instructions": "Which route best matches the input?", "criteria": {
        "accept": "The input clearly meets the requested condition.",
        "review": "More information is needed.",
        "ignore": "The input is not relevant.",
    }},
}


class DecisionConfig(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    studio_url: str = ""
    provider_id: str = "typesafe"
    questions: dict[str, Question] = msgspec.field(default_factory=lambda: msgspec.convert(DEFAULT_QUESTIONS, type=dict[str, Question]))
    route_question: str = "decision"
    min_confidence: float = 0.8
    min_probability: float = 0.8
    min_interval_ms: int = 100
    max_age_ms: int = 2000
    image_max_side: int = 768


@dataclass(frozen=True)
class PendingDecision:
    request: DecisionRequest
    exec_id: str | int
    received_at: float
    generation: int
    config: DecisionConfig


class DecisionRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(node_id=node_id, data_in_ports=["state", "video"],
                         data_out_ports=["answers", "value", "probabilities", "confidence", "probability", "accepted", "metrics", "error"],
                         state_fields=[field.name for field in node.stateFields or []],
                         exec_in_ports=["exec"], exec_out_ports=["decided", "uncertain", "error"])
        self._config = msgspec.convert(initial_state or {}, type=DecisionConfig)
        self._validate_config(self._config)
        self._active = True
        self._generation = 0
        self._pending: PendingDecision | None = None
        self._wake = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None
        self._video_reader: ZenohLatestVideoFrameTransport | None = None
        self._video_stream_key = ""
        self._zenoh_config_path: str | None = None
        self._zenoh_connect: tuple[str, ...] = ()
        self._zenoh_listen: tuple[str, ...] = ()
        self._zenoh_shm_pool_bytes = 256 * 1024 * 1024
        self._outputs: dict[str, F8JsonValue] = {}
        self._snapshots: OrderedDict[str | int, dict[str, F8JsonValue]] = OrderedDict()
        self._processed = 0
        self._dropped = 0
        self._failed = 0
        self._last_error = ""
        self._last_error_at = 0.0

    @staticmethod
    def _validate_config(config: DecisionConfig) -> None:
        if not config.provider_id.strip() or len(config.provider_id) > 128:
            raise ValueError("providerId must be a configured decision connection ID")
        validate_questions(config.questions)
        if config.route_question not in config.questions:
            raise ValueError("routeQuestion must name one of the configured questions")
        if not 0 <= config.min_confidence <= 1 or not 0.5 <= config.min_probability <= 1:
            raise ValueError("minConfidence must be in [0, 1] and minProbability in [0.5, 1]")
        if not 10 <= config.min_interval_ms <= 60000 or not 50 <= config.max_age_ms <= 60000:
            raise ValueError("minIntervalMs must be 10..60000 and maxAgeMs 50..60000")
        if not 64 <= config.image_max_side <= 2048:
            raise ValueError("imageMaxSide must be 64..2048")
        if config.studio_url:
            url = urlsplit(config.studio_url)
            if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("studioUrl must be an HTTP(S) server URL without credentials")

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        if isinstance(bus, ServiceBus):
            self._zenoh_config_path = bus.config.zenoh_config_path
            self._zenoh_connect = bus.config.zenoh_connect
            self._zenoh_listen = bus.config.zenoh_listen
            self._zenoh_shm_pool_bytes = bus.config.zenoh_shm_pool_bytes

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        current = cast(dict[str, F8JsonValue], msgspec.to_builtins(self._config))
        current[field] = unwrap_json_value(value)
        config = msgspec.convert(current, type=DecisionConfig)
        self._validate_config(config)
        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        current = cast(dict[str, F8JsonValue], msgspec.to_builtins(self._config))
        current[field] = unwrap_json_value(value)
        config = msgspec.convert(current, type=DecisionConfig)
        self._validate_config(config)
        self._config = config
        self._generation += 1
        pending, self._pending = self._pending, None
        if pending is not None:
            await self._discard(pending, "configuration or lifecycle changed")
        self._outputs["accepted"] = False
        await self.emit("accepted", False)

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self._active = active
        self._generation += 1
        pending, self._pending = self._pending, None
        if pending is not None:
            await self._discard(pending, "configuration or lifecycle changed")
        self._outputs["accepted"] = False
        await self.emit("accepted", False)

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        del in_port
        if not self._active:
            return []
        try:
            raw = unwrap_json_value(await self.pull("state", ctx_id=exec_id))
            video_key = self.input_zenoh_key("video")
            image_data_url = await asyncio.to_thread(self._capture_image, video_key, self._config.image_max_side) if video_key else None
            if video_key and image_data_url is None:
                raise ValueError("No video frame available for this decision trigger")
            request = msgspec.convert({"state": raw if raw is not None else {},
                                       "questions": msgspec.to_builtins(self._config.questions),
                                       "providerId": self._config.provider_id,
                                       "imageDataUrl": image_data_url}, type=DecisionRequest)
        except Exception as exc:
            self._failed += 1
            signature = f"{type(exc).__name__}: {exc}"
            now = time.monotonic()
            if signature != self._last_error or now - self._last_error_at >= 5:
                logger.exception("Decision input capture failed node=%s", self.node_id)
                self._last_error, self._last_error_at = signature, now
            pending = PendingDecision(DecisionRequest(state={}, questions=self._config.questions, provider_id=self._config.provider_id), exec_id, now, self._generation, self._config)
            await self._publish("accepted", False, pending)
            await self._publish("error", signature, pending)
            await self._metrics(pending)
            await self.emit_exec("error", exec_id=exec_id)
            return []
        if self._pending is not None:
            await self._discard(self._pending, "superseded by a newer trigger")
        self._pending = PendingDecision(request, exec_id, time.monotonic(), self._generation, self._config)
        if self._worker is None:
            self._client = httpx.AsyncClient(timeout=12, limits=httpx.Limits(max_connections=1, max_keepalive_connections=1))
            self._worker = asyncio.create_task(self._run(), name=f"decision:{self.node_id}")
            self._worker.add_done_callback(self._worker_finished)
        self._wake.set()
        return []

    def _capture_image(self, stream_key: str, max_side: int) -> str | None:
        if self._video_reader is None or self._video_stream_key != stream_key:
            if self._video_reader is not None:
                self._video_reader.close()
            self._video_reader = ZenohLatestVideoFrameTransport.open_subscriber(
                stream_key, config_path=self._zenoh_config_path, connect=self._zenoh_connect,
                listen=self._zenoh_listen, shm_pool_bytes=self._zenoh_shm_pool_bytes,
            )
            self._video_stream_key = stream_key
        frame = self._video_reader.wait_latest(500)
        if frame is None:
            return None
        with frame:
            if frame.fmt != VIDEO_FORMAT_BGRA32:
                raise ValueError("Decision video input requires BGRA32 frames")
            if frame.width <= 0 or frame.height <= 0 or frame.width > 8192 or frame.height > 8192 or frame.pitch < frame.width * 4:
                raise ValueError("Decision video frame dimensions or pitch are invalid")
            if frame.frame_bytes > 128 * 1024 * 1024:
                raise ValueError("Decision video frame exceeds 128 MB")
            image = Image.frombytes("RGBA", (frame.width, frame.height), frame.payload_bytes(), "raw", "BGRA", frame.pitch, 1)
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=78)
            data_url = "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")
            if len(data_url) > 2_000_000:
                raise ValueError("Decision image exceeds the 2 MB request limit")
            return data_url

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        if ctx_id is not None:
            snapshot = self._snapshots.get(ctx_id)
            return None if snapshot is None else snapshot.get(port)
        return self._outputs.get(port)

    def _worker_finished(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("Decision worker stopped node=%s", self.node_id, exc_info=(type(error), error, error.__traceback__))

    async def close(self) -> None:
        self._active = False
        self._pending = None
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._video_reader is not None:
            self._video_reader.close()
            self._video_reader = None

    async def _evaluate(self, pending: PendingDecision) -> DecisionResult:
        if self._client is None:
            raise RuntimeError("Decision HTTP client is not running")
        endpoint = pending.config.studio_url.strip() or os.environ.get("F8STUDIO_SERVER_URL", "http://127.0.0.1:8210")
        headers = {"Content-Type": "application/json"}
        token = os.environ.get("F8STUDIO_ACCESS_TOKEN", "")
        launched_url = os.environ.get("F8STUDIO_SERVER_URL", "")
        if token and urlsplit(endpoint).netloc == urlsplit(launched_url).netloc and urlsplit(endpoint).scheme == urlsplit(launched_url).scheme:
            headers["Authorization"] = f"Bearer {token}"
        response = await self._client.post(f"{endpoint.rstrip('/')}/api/decisions/evaluate", content=msgspec.json.encode(pending.request), headers=headers)
        response.raise_for_status()
        result = msgspec.json.decode(response.content, type=DecisionResult)
        validate_result(result, pending.request.questions)
        return result

    async def _publish(self, port: str, value: F8JsonValue, pending: PendingDecision) -> None:
        self._outputs[port] = value
        snapshot = self._snapshots.setdefault(pending.exec_id, {})
        snapshot[port] = value
        self._snapshots.move_to_end(pending.exec_id)
        while len(self._snapshots) > 64:
            self._snapshots.popitem(last=False)
        await self.emit(port, value, ctx_id=pending.exec_id)

    async def _metrics(self, pending: PendingDecision) -> None:
        await self._publish("metrics", {"processed": self._processed, "dropped": self._dropped, "failed": self._failed,
                                      "ageMs": (time.monotonic() - pending.received_at) * 1000, "requestId": str(pending.exec_id)}, pending)

    async def _discard(self, pending: PendingDecision, reason: str) -> None:
        self._dropped += 1
        await self._publish("accepted", False, pending)
        await self._publish("error", reason, pending)
        await self._metrics(pending)
        await self.emit_exec("error", exec_id=pending.exec_id)

    async def _run(self) -> None:
        next_request_at = 0.0
        while True:
            await self._wake.wait()
            await asyncio.sleep(max(0, next_request_at - time.monotonic()))
            self._wake.clear()
            pending, self._pending = self._pending, None
            if pending is None or not self._active:
                continue
            next_request_at = time.monotonic() + pending.config.min_interval_ms / 1000
            try:
                if (time.monotonic() - pending.received_at) * 1000 > pending.config.max_age_ms:
                    await self._discard(pending, "decision expired")
                    continue
                result = await self._evaluate(pending)
                if not self._active or pending.generation != self._generation:
                    await self._discard(pending, "configuration or lifecycle changed")
                    continue
                if (time.monotonic() - pending.received_at) * 1000 > pending.config.max_age_ms:
                    await self._discard(pending, "decision expired")
                    continue
                answer = result.answers[pending.config.route_question]
                probability: float | None = None
                confidence: float | None = None
                probabilities: dict[str, float] = {}
                value: F8JsonValue
                if isinstance(answer, ChoiceAnswer):
                    value = answer.choice
                    probabilities = answer.probabilities
                    probability = probabilities[answer.choice]
                    confidence = answer.confidence
                    accepted = confidence >= pending.config.min_confidence and probability >= pending.config.min_probability
                elif isinstance(answer, ScoreAnswer):
                    value = answer.score
                    probabilities = answer.probabilities
                    confidence = answer.confidence
                    accepted = confidence >= pending.config.min_confidence
                else:
                    probability = answer.noul
                    value = probability >= 0.5
                    accepted = max(probability, 1 - probability) >= pending.config.min_probability
                self._processed += 1
                await self._publish("answers", cast(F8JsonValue, msgspec.to_builtins(result)), pending)
                await self._publish("value", value, pending)
                await self._publish("probabilities", dict(probabilities), pending)
                await self._publish("probability", probability, pending)
                await self._publish("confidence", confidence, pending)
                await self._publish("accepted", accepted, pending)
                await self._publish("error", "", pending)
                await self._metrics(pending)
                await self.emit_exec("decided" if accepted else "uncertain", exec_id=pending.exec_id)
                self._last_error = ""
            except Exception as exc:
                # Background worker boundary: report once per error/interval, remain usable.
                self._failed += 1
                signature = f"{type(exc).__name__}: {exc}"
                now = time.monotonic()
                if signature != self._last_error or now - self._last_error_at >= 5:
                    logger.exception("Decision evaluation failed node=%s", self.node_id)
                    self._last_error, self._last_error_at = signature, now
                next_request_at = max(next_request_at, now + 2)
                if self._active and pending.generation == self._generation:
                    await self._publish("accepted", False, pending)
                    await self._publish("error", signature, pending)
                    await self._metrics(pending)
                    await self.emit_exec("error", exec_id=pending.exec_id)


DecisionRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1, serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS, version="0.0.1", label="Decision",
    description="Typed probabilistic decisions through a configured System-One host. Samples the latest video frame only on exec.",
    tags=["ai", "decision", "classification", "routing", "typesafe"],
    execInPorts=exec_port_specs(["exec"]), execOutPorts=exec_port_specs(["decided", "uncertain", "error"]),
    dataInPorts=[F8DataPortSpec(name="state", valueSchema=any_schema()), video_frame_port(name="video", description="Optional latest video frame, sampled on exec.")],
    dataOutPorts=[F8DataPortSpec(name=name, valueSchema=any_schema()) for name in ["answers", "value", "probabilities", "confidence", "probability", "metrics", "error"]] + [F8DataPortSpec(name="accepted", valueSchema=boolean_schema())],
    stateFields=[
        F8StateSpec(name="studioUrl", valueSchema=string_schema(default=""), access=F8StateAccess.rw, description="Empty uses the Studio server that launched this engine; standalone defaults to http://127.0.0.1:8210."),
        F8StateSpec(name="providerId", valueSchema=string_schema(default="typesafe"), access=F8StateAccess.rw, showOnNode=True, description="Connection ID of a System-One provider in Studio settings."),
        F8StateSpec(name="questions", valueSchema=F8AnyTypeSchema(default=DEFAULT_QUESTIONS), access=F8StateAccess.rw, description="Map of Choice, Score, or Noul questions evaluated together."),
        F8StateSpec(name="routeQuestion", valueSchema=string_schema(default="decision"), access=F8StateAccess.rw, showOnNode=True),
        F8StateSpec(name="minConfidence", valueSchema=number_schema(default=0.8, minimum=0, maximum=1), access=F8StateAccess.rw),
        F8StateSpec(name="minProbability", valueSchema=number_schema(default=0.8, minimum=0.5, maximum=1), access=F8StateAccess.rw),
        F8StateSpec(name="minIntervalMs", valueSchema=integer_schema(default=100, minimum=10, maximum=60000), access=F8StateAccess.rw),
        F8StateSpec(name="maxAgeMs", valueSchema=integer_schema(default=2000, minimum=50, maximum=60000), access=F8StateAccess.rw),
        F8StateSpec(name="imageMaxSide", valueSchema=integer_schema(default=768, minimum=64, maximum=2048), access=F8StateAccess.rw),
    ],
)


def register_operator(registry: Registry) -> None:
    registry.register_operator(DecisionRuntimeNode.SPEC, DecisionRuntimeNode, overwrite=True)
