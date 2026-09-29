from __future__ import annotations

from typing import Literal, cast
from urllib.parse import urlsplit

import httpx
import msgspec

from f8pysdk.decision import ChoiceQuestion, DecisionResult, validate_result

from .provider_settings import AgentProtocol, ModelCapabilities


class ProbeProviderRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel", forbid_unknown_fields=True):
    protocol: AgentProtocol
    endpoint: str
    api_key: str = ""
    provider_id: str = ""
    model: str = ""
    verify_model: bool = False


class ProviderProbeResult(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    connected: bool
    models: tuple[str, ...]
    detail: str
    verified: Literal["catalog", "model", "none"]
    model_capabilities: tuple[ModelCapabilities, ...] = ()


def _capability(entry: dict[str, object], name: str) -> bool | None:
    capabilities = entry.get("capabilities")
    if isinstance(capabilities, dict):
        value = cast(dict[str, object], capabilities).get(name)
        if isinstance(value, bool):
            return value
    return None


def _model_capabilities(entry: dict[str, object], model_id: str) -> ModelCapabilities:
    modalities = entry.get("input_modalities", entry.get("inputModalities"))
    architecture = entry.get("architecture")
    if modalities is None and isinstance(architecture, dict):
        architecture_data = cast(dict[str, object], architecture)
        modalities = architecture_data.get("input_modalities")
        if modalities is None:
            modality = architecture_data.get("modality")
            if isinstance(modality, str):
                modalities = modality.split("->", maxsplit=1)[0].split("+")
    image: bool | None = None
    if isinstance(modalities, list):
        input_modalities = cast(list[object], modalities)
        if all(isinstance(item, str) for item in input_modalities):
            image = "image" in input_modalities
    if image is None:
        image = _capability(entry, "vision")
    if image is None:
        image = _capability(entry, "image_input")
    thinking = _capability(entry, "reasoning")
    if thinking is None:
        thinking = _capability(entry, "thinking")
    if thinking is None:
        supported_parameters = entry.get("supported_parameters")
        if isinstance(supported_parameters, list):
            parameters = cast(list[object], supported_parameters)
            if all(isinstance(item, str) for item in parameters):
                thinking = "reasoning" in parameters or "reasoning_effort" in parameters
    return ModelCapabilities(model_id=model_id, image_input=image, thinking=thinking)


async def probe_provider(request: ProbeProviderRequest, *, saved_api_key: str = "", transport: httpx.AsyncBaseTransport | None = None) -> ProviderProbeResult:
    endpoint = request.endpoint.strip().rstrip("/")
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or any(character.isspace() for character in endpoint)):
        raise ValueError("Endpoint must be an HTTP(S) URL without credentials, query, or fragment")
    api_key = request.api_key.strip() or saved_api_key
    if request.protocol not in {"openai_chat", "systemone"} and not api_key:
        return ProviderProbeResult(connected=False, models=(), detail="API key is required", verified="none")
    headers = {"Accept": "application/json"}
    if request.protocol == "anthropic":
        headers.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=False, transport=transport) as client:
            if request.verify_model:
                if not request.model.strip():
                    raise ValueError("Select a model before testing inference")
                if request.protocol == "systemone":
                    path = "/systemone"
                    payload: dict[str, object] = {"model": request.model.strip(), "state": "connection test", "questions": {
                        "probe": {"type": "choice", "instructions": "Choose the matching label.",
                                  "criteria": {"connected": "The state is a connection test.", "other": "The state is something else."}},
                    }}
                elif request.protocol == "openai_responses":
                    path = "/responses"
                    payload = {"model": request.model.strip(), "input": "Reply OK.", "max_output_tokens": 32, "store": False}
                elif request.protocol == "anthropic":
                    path = "/messages"
                    payload = {"model": request.model.strip(), "max_tokens": 16, "messages": [{"role": "user", "content": "Reply OK."}]}
                else:
                    path = "/chat/completions"
                    payload = {"model": request.model.strip(), "max_tokens": 16, "messages": [{"role": "user", "content": "Reply OK."}]}
                response = await client.post(f"{endpoint}{path}", headers=headers, json=payload)
                if response.is_success:
                    try:
                        model_body = msgspec.json.decode(response.content, type=dict[str, object])
                    except msgspec.DecodeError:
                        return ProviderProbeResult(connected=False, models=(), detail="Model endpoint returned invalid JSON", verified="none")
                    if request.protocol == "systemone":
                        try:
                            result = msgspec.json.decode(response.content, type=DecisionResult)
                            validate_result(result, {"probe": ChoiceQuestion(
                                instructions="Choose the matching label.",
                                criteria={"connected": "The state is a connection test.", "other": "The state is something else."},
                            )})
                        except (msgspec.DecodeError, ValueError):
                            valid = False
                        else:
                            valid = True
                    elif request.protocol == "openai_responses":
                        valid = isinstance(model_body.get("output"), list)
                    elif request.protocol == "anthropic":
                        valid = model_body.get("type") == "message" and isinstance(model_body.get("content"), list)
                    else:
                        valid = isinstance(model_body.get("choices"), list)
                    if valid:
                        return ProviderProbeResult(connected=True, models=(request.model.strip(),), detail="Model inference succeeded", verified="model")
                    return ProviderProbeResult(connected=False, models=(), detail="Model endpoint returned an unexpected response", verified="none")
                return ProviderProbeResult(connected=False, models=(), detail=f"Model inference returned HTTP {response.status_code}", verified="none")

            response = await client.get(f"{endpoint}/models", headers=headers)
            if not response.is_success:
                return ProviderProbeResult(connected=False, models=(), detail=f"Model listing returned HTTP {response.status_code}; enter a model ID and test inference", verified="none")
            try:
                catalog_body = msgspec.json.decode(response.content, type=dict[str, object])
            except msgspec.DecodeError:
                return ProviderProbeResult(connected=False, models=(), detail="Model listing returned invalid JSON", verified="none")
            entries: object = catalog_body.get("data", catalog_body.get("models"))
            if not isinstance(entries, list):
                return ProviderProbeResult(connected=True, models=(), detail="API connected; model listing format is unavailable", verified="catalog")
            models_list: list[str] = []
            capabilities: list[ModelCapabilities] = []
            for entry in cast(list[object], entries):
                if not isinstance(entry, dict):
                    continue
                model_entry = cast(dict[str, object], entry)
                model_id: object = model_entry.get("id")
                if isinstance(model_id, str) and model_id not in models_list:
                    models_list.append(model_id)
                    capabilities.append(_model_capabilities(model_entry, model_id))
                if len(models_list) == 500:
                    break
            models = tuple(models_list)
            return ProviderProbeResult(connected=True, models=models, detail=f"API connected; found {len(models)} models", verified="catalog", model_capabilities=tuple(capabilities))
    except httpx.TimeoutException:
        return ProviderProbeResult(connected=False, models=(), detail="Connection timed out", verified="none")
    except httpx.RequestError as exc:
        return ProviderProbeResult(connected=False, models=(), detail=f"Connection failed: {type(exc).__name__}", verified="none")
