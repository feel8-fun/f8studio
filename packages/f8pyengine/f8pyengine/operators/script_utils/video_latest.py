"""Engine compatibility adapter; transport and decoding are owned by the SDK."""

from f8pysdk.video_latest import (
    VideoLatestConfig as VideoLatestConfig,
    VideoLatestPacketCodec as VideoLatestPacketCodec,
    VideoLatestSubscription as VideoLatestSubscription,
    VideoLatestSubscriptions as SharedVideoLatestSubscriptions,
    normalize_video_decode_mode as normalize_video_decode_mode,
)
from f8pysdk.video_transport import LatestVideoFrame


class VideoLatestSubscriptions(SharedVideoLatestSubscriptions):
    def _update_latest_packet(self, sub: VideoLatestSubscription, frame: LatestVideoFrame) -> bool:
        # Engine scripts historically reject invalid geometry even in raw mode.
        if frame.width <= 0 or frame.height <= 0 or frame.pitch <= 0:
            return False
        return super()._update_latest_packet(sub, frame)
