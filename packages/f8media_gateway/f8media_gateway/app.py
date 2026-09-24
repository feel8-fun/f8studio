from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TypeVar, cast

import msgspec
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

from f8pysdk.specs import F8JsonValue

from f8media_protocol.contracts import MediaGateway
from f8media_protocol.models import AudioSessionOffer, MediaSessionOffer, OverlayResult

from .service import InProcessMediaGateway, MEDIA_GATEWAY_VERSION


logger = logging.getLogger(__name__)
T = TypeVar("T")


def _json_value(value: object) -> F8JsonValue:
    return cast(F8JsonValue, msgspec.to_builtins(value, str_keys=True))


async def _decode_body(request: Request, value_type: type[T]) -> T:
    try:
        return msgspec.json.decode(await request.body(), type=value_type)
    except msgspec.DecodeError as exc:
        raise HTTPException(status_code=422, detail=f"invalid request body: {exc}") from exc


def create_app(*, gateway: MediaGateway | None = None) -> FastAPI:
    media_gateway = gateway or InProcessMediaGateway()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        await media_gateway.start()
        try:
            yield
        finally:
            await media_gateway.close()

    app = FastAPI(
        title="Feel8 Media Gateway API",
        version=MEDIA_GATEWAY_VERSION,
        docs_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "::1", "testserver"],
    )

    @app.exception_handler(ValueError)
    async def invalid_value(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unhandled_error(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled Media Gateway API error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": {"code": "internal_error", "message": f"{type(exc).__name__}: {exc}"}},
        )

    @app.get("/api/health")
    async def health() -> F8JsonValue:
        return _json_value(await media_gateway.health())

    @app.post("/api/media/sessions", status_code=201)
    async def create_video_session(request: Request) -> F8JsonValue:
        return _json_value(await media_gateway.create_video_session(await _decode_body(request, MediaSessionOffer)))

    @app.delete("/api/media/sessions/{session_id}", status_code=204)
    async def close_video_session(session_id: str) -> Response:
        if not await media_gateway.close_video_session(session_id):
            raise HTTPException(status_code=404, detail="media session not found")
        return Response(status_code=204)

    @app.post("/api/audio/sessions", status_code=201)
    async def create_audio_session(request: Request) -> F8JsonValue:
        return _json_value(await media_gateway.create_audio_session(await _decode_body(request, AudioSessionOffer)))

    @app.delete("/api/audio/sessions/{session_id}", status_code=204)
    async def close_audio_session(session_id: str) -> Response:
        if not await media_gateway.close_audio_session(session_id):
            raise HTTPException(status_code=404, detail="audio session not found")
        return Response(status_code=204)

    @app.post("/api/media/overlays", status_code=202)
    async def publish_overlay(request: Request) -> F8JsonValue:
        await media_gateway.publish_overlay(await _decode_body(request, OverlayResult))
        return {"accepted": True}

    @app.get("/api/media/sessions/{session_id}/media-timestamps/{media_timestamp}")
    async def frame_mapping(session_id: str, media_timestamp: int) -> F8JsonValue:
        mapping = await media_gateway.frame_mapping(session_id, media_timestamp)
        if mapping is None:
            raise HTTPException(status_code=404, detail="media frame mapping not found")
        return _json_value(mapping)

    @app.get("/api/media/sample")
    async def sample(source: str, x: int, y: int) -> F8JsonValue:
        try:
            return _json_value(await media_gateway.sample(source, x=x, y=y))
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=f"media source did not produce a frame: {source}") from exc

    @app.get("/api/media/metrics")
    async def metrics() -> F8JsonValue:
        return _json_value(await media_gateway.metrics())

    return app


__all__ = ["create_app"]
