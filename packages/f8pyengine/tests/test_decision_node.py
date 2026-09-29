from __future__ import annotations

import asyncio
import base64
import io
from typing import Any

import msgspec
import pytest
from PIL import Image

from f8pysdk.decision import DecisionResult
from f8pysdk.specs import F8RuntimeNode
from f8pysdk.video_transport import LatestVideoFrame, VIDEO_FORMAT_BGRA32
from f8pyengine.operators.decision import DecisionRuntimeNode, PendingDecision


class ControlledDecision(DecisionRuntimeNode):
    def __init__(self, *, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(node_id="decision", node=F8RuntimeNode(nodeId="decision", serviceId="engine", serviceClass="f8.pyengine", operatorClass="f8.decision", stateFields=list(self.SPEC.stateFields)), initial_state=initial_state)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.completed: asyncio.Queue[str] = asyncio.Queue()
        self.calls: list[str | int] = []
        self.failure = False
        self.answer: dict[str, Any] = {"type": "choice", "choice": "accept", "probabilities": {"accept": 0.95, "review": 0.03, "ignore": 0.02}, "confidence": 0.925}

    async def pull(self, port: str, *, ctx_id: str | int | None = None) -> Any:
        return f"input-{ctx_id}"

    async def _evaluate(self, pending: PendingDecision) -> DecisionResult:
        self.calls.append(pending.exec_id)
        self.started.set()
        await self.release.wait()
        if self.failure:
            raise ValueError("Invalid upstream result")
        return msgspec.convert({"model": "jev-test", "answers": {"decision": self.answer}, "usage": {"input_tokens": 10, "output_tokens": 0}}, type=DecisionResult)

    async def emit_exec(self, port: str, *, exec_id: str | int) -> None:
        await self.completed.put(port)


def test_latest_pending_input_replaces_backlog_and_preserves_request_ids() -> None:
    async def scenario() -> None:
        node = ControlledDecision(initial_state={"minIntervalMs": 10})
        try:
            await node.on_exec(1)
            await asyncio.wait_for(node.started.wait(), 1)
            for index in range(2, 11):
                await node.on_exec(index)
            node.release.set()
            assert await asyncio.wait_for(node.completed.get(), 1) == "decided"
            assert await asyncio.wait_for(node.completed.get(), 1) == "decided"
            assert node.calls == [1, 10]
            assert await node.compute_output("accepted") is True
            metrics = await node.compute_output("metrics")
            assert metrics["dropped"] == 8 and metrics["processed"] == 2 and metrics["requestId"] == "10"
            first_metrics = await node.compute_output("metrics", ctx_id=1)
            assert first_metrics["requestId"] == "1" and first_metrics["processed"] == 1
            assert await node.compute_output("value", ctx_id="unknown-request") is None
        finally:
            await node.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("probability,confidence,route", [(0.95, 0.9, "decided"), (0.95, 0.4, "uncertain"), (0.6, 0.9, "uncertain")])
def test_choice_routing_checks_probability_and_confidence_separately(probability: float, confidence: float, route: str) -> None:
    async def scenario() -> None:
        node = ControlledDecision()
        node.answer = {"type": "choice", "choice": "accept", "probabilities": {"accept": probability, "review": 1 - probability, "ignore": 0}, "confidence": confidence}
        node.release.set()
        try:
            await node.on_exec(1)
            assert await asyncio.wait_for(node.completed.get(), 1) == route
            assert await node.compute_output("probability") == probability
            assert await node.compute_output("confidence") == confidence
        finally:
            await node.close()
    asyncio.run(scenario())


def test_noul_keeps_yes_probability_and_does_not_invent_confidence() -> None:
    async def scenario() -> None:
        node = ControlledDecision(initial_state={"questions": {"decision": {"type": "noul", "instructions": "Is it relevant?"}}})
        node.answer = {"type": "noul", "noul": 0.05}
        node.release.set()
        try:
            await node.on_exec(1)
            assert await asyncio.wait_for(node.completed.get(), 1) == "decided"
            assert await node.compute_output("value") is False
            assert await node.compute_output("probability") == 0.05
            assert await node.compute_output("confidence") is None
        finally:
            await node.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("invalidate", ["pause", "config", "age"])
def test_obsolete_results_never_trigger_a_branch(invalidate: str) -> None:
    async def scenario() -> None:
        node = ControlledDecision(initial_state={"maxAgeMs": 50})
        try:
            await node.on_exec(1)
            await asyncio.wait_for(node.started.wait(), 1)
            if invalidate == "pause":
                await node.on_lifecycle(False, {})
            elif invalidate == "config":
                await node.on_state("minConfidence", 0.99)
            else:
                await asyncio.sleep(0.06)
            node.release.set()
            await asyncio.sleep(0.02)
            assert node.completed.empty()
        finally:
            await node.close()
    asyncio.run(scenario())


def test_failure_produces_an_error_output_and_never_a_success_branch() -> None:
    async def scenario() -> None:
        node = ControlledDecision()
        node.failure = True
        node.release.set()
        try:
            await node.on_exec(1)
            assert await asyncio.wait_for(node.completed.get(), 1) == "error"
            assert await node.compute_output("accepted") is False
            assert "Invalid upstream result" in await node.compute_output("error")
        finally:
            await node.close()
    asyncio.run(scenario())


def test_video_is_sampled_only_when_exec_fires() -> None:
    class VideoDecision(ControlledDecision):
        def __init__(self) -> None:
            super().__init__(initial_state={"providerId": "systemone_local"})
            self.samples = 0

        def input_zenoh_key(self, port: str) -> str | None:
            assert port == "video"
            return "f8/test/video"

        def _capture_image(self, stream_key: str, max_side: int) -> str | None:
            assert stream_key == "f8/test/video" and max_side == 768
            self.samples += 1
            return "data:image/jpeg;base64,/9j/"

    async def scenario() -> None:
        node = VideoDecision()
        node.release.set()
        try:
            await node.on_data("video", {"frameId": 1})
            assert node.samples == 0
            await node.on_exec(1)
            assert await asyncio.wait_for(node.completed.get(), 1) == "decided"
            assert node.samples == 1
        finally:
            await node.close()

    asyncio.run(scenario())


def test_missing_video_frame_routes_to_error() -> None:
    class MissingVideo(ControlledDecision):
        def input_zenoh_key(self, port: str) -> str | None:
            return "f8/test/video"

        def _capture_image(self, stream_key: str, max_side: int) -> str | None:
            return None

    async def scenario() -> None:
        node = MissingVideo()
        try:
            await node.on_exec(1)
            assert await asyncio.wait_for(node.completed.get(), 1) == "error"
            assert "No video frame available" in await node.compute_output("error", ctx_id=1)
            assert node.calls == []
        finally:
            await node.close()

    asyncio.run(scenario())


def test_video_frame_is_encoded_as_bounded_jpeg() -> None:
    class FrameReader:
        def wait_latest(self, timeout_ms: int) -> LatestVideoFrame:
            assert timeout_ms == 500
            return LatestVideoFrame(width=2, height=1, pitch=8, fmt=VIDEO_FORMAT_BGRA32,
                                    frame_id=1, ts_ms=0, payload=memoryview(bytes([0, 0, 255, 255, 0, 255, 0, 255])))

        def close(self) -> None:
            pass

    node = ControlledDecision()
    node._video_reader = FrameReader()  # type: ignore[assignment]
    node._video_stream_key = "f8/test/video"
    data_url = node._capture_image("f8/test/video", 64)
    assert data_url is not None and data_url.startswith("data:image/jpeg;base64,")
    with Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as image:
        assert image.size == (2, 1) and image.format == "JPEG"
