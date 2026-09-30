from __future__ import annotations

from f8studio_server.agents.provider_settings import ModelCapabilities

import asyncio
import json
from pathlib import Path

import httpx
import msgspec
import pytest

from f8pysdk.decision import DecisionRequest, DecisionResult, validate_questions, validate_result
from f8studio_server.agents.decisions import DecisionCapacityError, DecisionResponseError, SystemOneDecisionClient
from f8studio_server.agents.provider_settings import CreateProviderConnection, ModelCapabilities, UpdateProviderSettings
from f8studio_server.agents.providers import AgentProviderRegistry


QUESTIONS = {
    "route": {"type": "choice", "instructions": "Choose a route", "criteria": {"yes": "relevant", "no": "irrelevant"}},
    "severity": {"type": "score", "instructions": "Rate severity", "criteria": ["low", "high"]},
    "motion": {"type": "noul", "instructions": "Is there motion?"},
}
RESULT = {
    "model": "jev-1.13.0", "answers": {
        "route": {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.9, "no": 0.1}, "confidence": 0.8},
        "severity": {"type": "score", "score": 0.75, "legend": {"0": "low", "1": "high"}, "probabilities": {"0": 0.25, "1": 0.75}, "confidence": 0.5},
        "motion": {"type": "noul", "noul": 0.9},
    }, "usage": {"input_tokens": 100, "output_tokens": 20},
}


def registry(tmp_path: Path) -> AgentProviderRegistry:
    providers = AgentProviderRegistry(tmp_path / "providers.json")
    providers.update_settings("typesafe", UpdateProviderSettings(model="jev-latest", endpoint="https://api.typesafe.ai/v1", api_key="test-secret"))
    return providers


def test_typesafe_is_a_decision_provider_not_a_chat_agent(tmp_path: Path) -> None:
    providers = registry(tmp_path)
    view = next(setting for setting in providers.settings() if setting.provider_id == "typesafe")
    assert view.kind == "decision" and view.input_modalities == ("text",)
    assert "typesafe" not in {summary.provider_id for summary in providers.summaries()}
    assert b"test-secret" not in msgspec.json.encode(view)


def test_system_one_wire_contract_batches_questions_and_preserves_probability_semantics(tmp_path: Path) -> None:
    async def scenario() -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
            assert request.headers["Authorization"] == "Bearer test-secret"
            body = json.loads(request.content)
            assert body["model"] == "jev-latest"
            assert body["state"] == {"observation": "motion detected"}
            assert set(body["questions"]) == set(QUESTIONS)
            assert "messages" not in body
            return httpx.Response(200, json=RESULT)
        client = SystemOneDecisionClient(registry(tmp_path), transport=httpx.MockTransport(respond))
        try:
            request = msgspec.convert({"state": {"observation": "motion detected"}, "questions": QUESTIONS}, type=DecisionRequest)
            result = await client.evaluate(request)
            assert msgspec.to_builtins(result) == RESULT
        finally:
            await client.close()
    asyncio.run(scenario())


def test_local_system_one_accepts_text_without_key_and_jpeg_when_enabled(tmp_path: Path) -> None:
    async def scenario() -> None:
        providers = registry(tmp_path)
        providers.update_settings("systemone_local", UpdateProviderSettings(
            model="vjev-vision", endpoint="http://127.0.0.1:8001/v1",
            model_capabilities=(ModelCapabilities(model_id="vjev-vision", image_input=True),),
        ))
        bodies: list[dict[str, object]] = []

        def respond(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == "http://127.0.0.1:8001/v1/systemone"
            assert "Authorization" not in request.headers
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json=RESULT)

        client = SystemOneDecisionClient(providers, transport=httpx.MockTransport(respond))
        try:
            base = {"state": {"observation": "motion detected"}, "questions": QUESTIONS, "providerId": "systemone_local"}
            await client.evaluate(msgspec.convert(base, type=DecisionRequest))
            await client.evaluate(msgspec.convert({**base, "imageDataUrl": "data:image/jpeg;base64,/9j/"}, type=DecisionRequest))
            assert bodies[0]["state"] == {"observation": "motion detected"}
            assert bodies[1]["state"] == {"observation": "motion detected", "image": "data:image/jpeg;base64,/9j/"}
            assert all(body["model"] == "vjev-vision" for body in bodies)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_named_systemone_connection_routes_vision_by_model(tmp_path: Path) -> None:
    async def scenario() -> None:
        providers = AgentProviderRegistry(tmp_path / "providers.json")
        connection = providers.create_connection(CreateProviderConnection(
            display_name="Vision host", protocol="systemone", endpoint="http://localhost:8002/v1",
            model="vision", models=("vision",), model_capabilities=(ModelCapabilities(model_id="vision", image_input=True),),
        ))
        def respond(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == "http://localhost:8002/v1/systemone"
            assert json.loads(request.content)["state"]["image"] == "data:image/jpeg;base64,/9j/"
            return httpx.Response(200, json=RESULT)
        client = SystemOneDecisionClient(providers, transport=httpx.MockTransport(respond))
        try:
            request = msgspec.convert({"state": "frame", "questions": QUESTIONS,
                                       "providerId": connection.provider_id,
                                       "imageDataUrl": "data:image/jpeg;base64,/9j/"}, type=DecisionRequest)
            assert (await client.evaluate(request)).model == "jev-1.13.0"
        finally:
            await client.close()
    asyncio.run(scenario())


def test_text_only_provider_rejects_images_before_upstream_call(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = SystemOneDecisionClient(registry(tmp_path), transport=httpx.MockTransport(
            lambda _: pytest.fail("Image request must not reach a text-only provider")
        ))
        try:
            request = msgspec.convert({"state": "input", "questions": QUESTIONS,
                                       "imageDataUrl": "data:image/jpeg;base64,/9j/"}, type=DecisionRequest)
            with pytest.raises(ValueError, match="does not support image"):
                await client.evaluate(request)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_invalid_remote_results_are_rejected(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = SystemOneDecisionClient(registry(tmp_path), transport=httpx.MockTransport(lambda _: httpx.Response(200, json={**RESULT, "answers": {}})))
        try:
            request = msgspec.convert({"state": "input", "questions": QUESTIONS}, type=DecisionRequest)
            with pytest.raises(DecisionResponseError):
                await client.evaluate(request)
        finally:
            await client.close()
    asyncio.run(scenario())


def test_concurrency_is_bounded_without_an_unbounded_waiting_queue(tmp_path: Path) -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        full = asyncio.Event()
        count = 0
        async def respond(_: httpx.Request) -> httpx.Response:
            nonlocal count
            count += 1
            if count == 4:
                full.set()
            await release.wait()
            return httpx.Response(200, json=RESULT)
        client = SystemOneDecisionClient(registry(tmp_path), transport=httpx.MockTransport(respond))
        request = msgspec.convert({"state": "input", "questions": QUESTIONS}, type=DecisionRequest)
        tasks = [asyncio.create_task(client.evaluate(request)) for _ in range(4)]
        try:
            await asyncio.wait_for(full.wait(), 1)
            with pytest.raises(DecisionCapacityError):
                await client.evaluate(request)
        finally:
            release.set()
            await asyncio.gather(*tasks)
            await client.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("probabilities", [{"yes": 0.9, "other": 0.1}, {"yes": 0.9, "no": 0.9}])
def test_probability_distributions_match_candidates_and_sum_to_one(probabilities: dict[str, float]) -> None:
    request = msgspec.convert({"state": "input", "questions": {"route": QUESTIONS["route"]}}, type=DecisionRequest)
    result = msgspec.convert({"model": "jev", "usage": RESULT["usage"], "answers": {"route": {"type": "choice", "choice": "yes", "probabilities": probabilities, "confidence": 0.8}}}, type=DecisionResult)
    with pytest.raises(ValueError):
        validate_result(result, request.questions)


def test_question_validation_rejects_empty_and_invalid_rubrics() -> None:
    request = msgspec.convert({"state": "input", "questions": {"score": {"type": "score", "instructions": "Rate", "criteria": ["only"]}}}, type=DecisionRequest)
    with pytest.raises(ValueError, match="2 to 10"):
        validate_questions(request.questions)
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert({"state": "input", "images": ["image.png"], "questions": QUESTIONS}, type=DecisionRequest)
