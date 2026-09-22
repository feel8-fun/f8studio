from __future__ import annotations

from typing import Protocol

from .models import (
    AudioSessionAnswer,
    AudioSessionOffer,
    MediaFrameMapping,
    MediaGatewayHealth,
    MediaMetrics,
    MediaSample,
    MediaSessionAnswer,
    MediaSessionOffer,
    OverlayResult,
)


class MediaGateway(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def health(self) -> MediaGatewayHealth: ...

    async def create_video_session(self, offer: MediaSessionOffer) -> MediaSessionAnswer: ...

    async def close_video_session(self, session_id: str) -> bool: ...

    async def create_audio_session(self, offer: AudioSessionOffer) -> AudioSessionAnswer: ...

    async def close_audio_session(self, session_id: str) -> bool: ...

    async def publish_overlay(self, result: OverlayResult) -> None: ...

    async def frame_mapping(self, session_id: str, media_timestamp: int) -> MediaFrameMapping | None: ...

    async def sample(self, source: str, *, x: int, y: int) -> MediaSample: ...

    async def metrics(self) -> MediaMetrics: ...


__all__ = ["MediaGateway"]
