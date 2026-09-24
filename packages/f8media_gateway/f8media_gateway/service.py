from __future__ import annotations

import os
from uuid import uuid4

from f8media_protocol.models import (
    MEDIA_API_VERSION,
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

from .audio_media import AudioSessionManager
from .media import MediaSessionManager


MEDIA_GATEWAY_VERSION = "0.1.0"


class InProcessMediaGateway:
    def __init__(
        self,
        *,
        video: MediaSessionManager | None = None,
        audio: AudioSessionManager | None = None,
        gateway_epoch: str | None = None,
    ) -> None:
        self.video = video or MediaSessionManager()
        self.audio = audio or AudioSessionManager()
        self._gateway_epoch = gateway_epoch or uuid4().hex

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        await self.video.close()
        await self.audio.close()

    async def health(self) -> MediaGatewayHealth:
        return MediaGatewayHealth(
            status="ok",
            service="f8media-gateway",
            version=MEDIA_GATEWAY_VERSION,
            protocol_version=MEDIA_API_VERSION,
            gateway_epoch=self._gateway_epoch,
            process_id=os.getpid(),
        )

    async def create_video_session(self, offer: MediaSessionOffer) -> MediaSessionAnswer:
        return await self.video.create(offer)

    async def close_video_session(self, session_id: str) -> bool:
        return await self.video.close_session(session_id)

    async def create_audio_session(self, offer: AudioSessionOffer) -> AudioSessionAnswer:
        return await self.audio.create(offer)

    async def close_audio_session(self, session_id: str) -> bool:
        return await self.audio.close_session(session_id)

    async def publish_overlay(self, result: OverlayResult) -> None:
        await self.video.publish_overlay(result)

    async def frame_mapping(self, session_id: str, media_timestamp: int) -> MediaFrameMapping | None:
        return await self.video.frame_mapping(session_id, media_timestamp)

    async def sample(self, source: str, *, x: int, y: int) -> MediaSample:
        return await self.video.sample(source, x=x, y=y)

    async def metrics(self) -> MediaMetrics:
        overlay = self.video.overlays.counters
        return MediaMetrics(
            video_sessions=self.video.session_count,
            video_sources=self.video.source_count,
            audio_sessions=self.audio.session_count,
            audio_sources=self.audio.source_count,
            overlay_matched=overlay.matched,
            overlay_wait_timeouts=overlay.wait_timeouts,
            overlay_expired=overlay.expired,
            overlay_rejected=overlay.rejected,
            audio_dropped_chunks=self.audio.dropped_chunks,
        )


__all__ = ["InProcessMediaGateway", "MEDIA_GATEWAY_VERSION"]
