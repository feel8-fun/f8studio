from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import TypeVar, overload
from urllib.parse import quote

import httpx
import msgspec

from f8pysdk.specs import F8JsonValue

from .models import (
    MEDIA_API_VERSION,
    AudioSessionAnswer,
    AudioSessionOffer,
    MediaErrorResponse,
    MediaFrameMapping,
    MediaGatewayHealth,
    MediaMetrics,
    MediaSample,
    MediaSessionAnswer,
    MediaSessionOffer,
    OverlayResult,
)


logger = logging.getLogger(__name__)
T = TypeVar("T")


class MediaGatewayUnavailable(RuntimeError):
    pass


class MediaGatewayRequestError(RuntimeError):
    def __init__(self, status_code: int, detail: F8JsonValue) -> None:
        super().__init__(f"Media Gateway request failed with HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class RemoteMediaGatewayConfig:
    base_url: str = "http://127.0.0.1:8211"
    manage_process: bool = True
    startup_timeout_s: float = 10.0


class RemoteMediaGateway:
    def __init__(
        self,
        config: RemoteMediaGatewayConfig | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config or RemoteMediaGatewayConfig()
        self._client = client
        self._owns_client = client is None
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._config.base_url, timeout=5.0)
        try:
            if self._config.manage_process:
                if self._process is not None:
                    raise RuntimeError("Media Gateway process is already running")
                host, port = _loopback_endpoint(self._config.base_url)
                self._process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "f8media_gateway",
                    "--host",
                    host,
                    "--port",
                    str(port),
                    "--exit-on-stdin-close",
                    stdin=asyncio.subprocess.PIPE,
                )
            await self._wait_until_ready()
        except Exception:
            logger.exception("Media Gateway failed to start base_url=%s", self._config.base_url)
            await self.close()
            raise

    async def _wait_until_ready(self) -> None:
        deadline = asyncio.get_running_loop().time() + self._config.startup_timeout_s
        last_error = "gateway did not respond"
        while asyncio.get_running_loop().time() < deadline:
            process = self._process
            if process is not None and process.returncode is not None:
                raise MediaGatewayUnavailable(f"Media Gateway exited during startup with code {process.returncode}")
            try:
                health = await self.health()
            except MediaGatewayUnavailable as exc:
                last_error = str(exc)
            else:
                if health.service != "f8media-gateway":
                    raise MediaGatewayUnavailable(
                        "Media Gateway service mismatch: "
                        f"expected 'f8media-gateway', received {health.service!r}"
                    )
                if health.protocol_version != MEDIA_API_VERSION:
                    raise MediaGatewayUnavailable(
                        "Media Gateway protocol mismatch: "
                        f"expected {MEDIA_API_VERSION}, received {health.protocol_version}"
                    )
                if process is not None and health.process_id != process.pid:
                    raise MediaGatewayUnavailable(
                        "Media Gateway process mismatch: "
                        f"started PID {process.pid}, health reported PID {health.process_id}"
                    )
                if health.status == "ok":
                    return
                last_error = f"unexpected health status {health.status!r}"
            await asyncio.sleep(0.05)
        raise MediaGatewayUnavailable(f"Media Gateway startup timed out: {last_error}")

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None and self._owns_client:
            await client.aclose()
        process = self._process
        self._process = None
        if process is None or process.returncode is not None:
            return
        if process.stdin is not None:
            process.stdin.close()
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except TimeoutError:
            process.kill()
            await process.wait()

    async def health(self) -> MediaGatewayHealth:
        return await self._request("GET", "/api/health", response_type=MediaGatewayHealth)

    async def create_video_session(self, offer: MediaSessionOffer) -> MediaSessionAnswer:
        return await self._request(
            "POST", "/api/media/sessions", body=offer, response_type=MediaSessionAnswer
        )

    async def close_video_session(self, session_id: str) -> bool:
        return await self._delete_session(f"/api/media/sessions/{quote(session_id, safe='')}")

    async def create_audio_session(self, offer: AudioSessionOffer) -> AudioSessionAnswer:
        return await self._request(
            "POST", "/api/audio/sessions", body=offer, response_type=AudioSessionAnswer
        )

    async def close_audio_session(self, session_id: str) -> bool:
        return await self._delete_session(f"/api/audio/sessions/{quote(session_id, safe='')}")

    async def publish_overlay(self, result: OverlayResult) -> None:
        await self._request("POST", "/api/media/overlays", body=result, response_type=None)

    async def frame_mapping(self, session_id: str, media_timestamp: int) -> MediaFrameMapping | None:
        try:
            return await self._request(
                "GET",
                f"/api/media/sessions/{quote(session_id, safe='')}/media-timestamps/{media_timestamp}",
                response_type=MediaFrameMapping,
            )
        except MediaGatewayRequestError as exc:
            if exc.status_code == 404:
                return None
            raise

    async def sample(self, source: str, *, x: int, y: int) -> MediaSample:
        return await self._request(
            "GET",
            "/api/media/sample",
            query={"source": source, "x": str(x), "y": str(y)},
            response_type=MediaSample,
        )

    async def metrics(self) -> MediaMetrics:
        return await self._request("GET", "/api/media/metrics", response_type=MediaMetrics)

    async def _delete_session(self, path: str) -> bool:
        try:
            await self._request("DELETE", path, response_type=None)
            return True
        except MediaGatewayRequestError as exc:
            if exc.status_code == 404:
                return False
            raise

    @overload
    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: object | None = None,
        query: dict[str, str] | None = None,
        response_type: type[T],
    ) -> T: ...

    @overload
    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: object | None = None,
        query: dict[str, str] | None = None,
        response_type: None,
    ) -> None: ...

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: object | None = None,
        query: dict[str, str] | None = None,
        response_type: type[T] | None,
    ) -> T | None:
        client = self._client
        if client is None:
            raise MediaGatewayUnavailable("Media Gateway client is not started")
        try:
            response = await client.request(
                method,
                path,
                content=None if body is None else msgspec.json.encode(body),
                params=query,
                headers=None if body is None else {"content-type": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise MediaGatewayUnavailable(f"Media Gateway request failed: {type(exc).__name__}: {exc}") from exc
        if response.status_code >= 400:
            try:
                detail = msgspec.json.decode(response.content, type=MediaErrorResponse).detail
            except msgspec.DecodeError as exc:
                raise MediaGatewayUnavailable(
                    f"Media Gateway returned malformed HTTP {response.status_code} error"
                ) from exc
            raise MediaGatewayRequestError(response.status_code, detail)
        if response_type is None:
            return None
        try:
            return msgspec.json.decode(response.content, type=response_type)
        except msgspec.DecodeError as exc:
            raise MediaGatewayUnavailable(f"Media Gateway returned malformed {response_type.__name__}") from exc


def _loopback_endpoint(base_url: str) -> tuple[str, int]:
    parsed = httpx.URL(base_url)
    if parsed.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Media Gateway URL must use a loopback host")
    if parsed.scheme != "http":
        raise ValueError("Media Gateway URL must use HTTP")
    if parsed.path not in {"", "/"} or parsed.query:
        raise ValueError("Media Gateway URL must not contain a path or query")
    host = "::1" if parsed.host == "::1" else "127.0.0.1"
    return host, parsed.port or 80


__all__ = [
    "MediaGatewayRequestError",
    "MediaGatewayUnavailable",
    "RemoteMediaGateway",
    "RemoteMediaGatewayConfig",
]
