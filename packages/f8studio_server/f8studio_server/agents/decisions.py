from __future__ import annotations

import asyncio
import logging
import time

import httpx
import msgspec

from f8pysdk.decision import DecisionRequest, DecisionResult, validate_questions, validate_result
from f8pysdk.specs import F8JsonValue

from .providers import AgentProviderRegistry

logger = logging.getLogger(__name__)


class DecisionCapacityError(RuntimeError):
    pass


class DecisionResponseError(RuntimeError):
    pass


class SystemOneDecisionClient:
    def __init__(self, providers: AgentProviderRegistry, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._providers = providers
        self._slots = asyncio.Semaphore(4)
        self._last_error: str = ""
        self._last_error_at = 0.0
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(10, connect=5),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
            transport=transport,
        )

    async def evaluate(self, request: DecisionRequest) -> DecisionResult:
        validate_questions(request.questions)
        config = self._providers.decision_config(request.provider_id)
        state: F8JsonValue = request.state
        if request.image_data_url is not None:
            if not self._providers.supports_image(request.provider_id, config.model):
                raise ValueError("Selected decision provider does not support image input")
            image_url = request.image_data_url
            if not image_url.startswith("data:image/jpeg;base64,") or len(image_url) > 2_000_000:
                raise ValueError("Decision image must be a JPEG data URL smaller than 2 MB")
            if isinstance(state, dict):
                if "image" in state:
                    raise ValueError("State already contains an image field")
                state = {**state, "image": image_url}
            else:
                state = {"text": state, "image": image_url}
        if self._slots.locked():
            raise DecisionCapacityError("All decision request slots are busy; retry with the latest input")
        async with self._slots:
            headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
            response = await self._http.post(
                f"{config.endpoint}/systemone",
                headers=headers,
                json={"model": config.model, "state": state, "questions": msgspec.to_builtins(request.questions)},
            )
            response.raise_for_status()
            try:
                result = msgspec.json.decode(response.content, type=DecisionResult)
                validate_result(result, request.questions)
            except (msgspec.DecodeError, ValueError) as exc:
                raise DecisionResponseError("System-One host returned an invalid decision response") from exc
            return result

    async def close(self) -> None:
        await self._http.aclose()

    def report_failure(self, message: str, error: Exception) -> None:
        now = time.monotonic()
        if message != self._last_error or now - self._last_error_at >= 5:
            logger.error(message, exc_info=(type(error), error, error.__traceback__))
            self._last_error, self._last_error_at = message, now
