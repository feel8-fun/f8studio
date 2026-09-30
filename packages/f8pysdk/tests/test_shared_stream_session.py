from unittest.mock import patch

from f8pysdk.binary_stream_transport import SharedStreamSession


class Subscriber:
    def __init__(self) -> None:
        self.closed = False

    def undeclare(self) -> None:
        self.closed = True


class Session:
    def __init__(self) -> None:
        self.subscribers: list[Subscriber] = []
        self.closed = False

    def declare_subscriber(self, key, callback) -> Subscriber:
        subscriber = Subscriber()
        self.subscribers.append(subscriber)
        return subscriber

    def close(self) -> None:
        self.closed = True


def test_video_and_audio_subscribers_borrow_one_owned_session() -> None:
    session = Session()
    with patch("f8pysdk.binary_stream_transport._open_zenoh_stream_session", return_value=session) as opened:
        shared = SharedStreamSession()
        video = shared.subscribe("f8/video", log_context="video")
        audio = shared.subscribe("f8/audio", log_context="audio", max_pending_samples=16)
        assert opened.call_count == 1
        video.close()
        assert session.subscribers[0].closed
        assert not session.subscribers[1].closed
        assert not session.closed
        audio.close()
        shared.close()
        assert session.closed
