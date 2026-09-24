from __future__ import annotations

from dataclasses import dataclass

from f8pysdk.binary_stream_transport import ZenohLatestBinaryStreamTransport


class _Session:
    def close(self) -> None:
        return


@dataclass(frozen=True)
class _Sample:
    payload: object


class _CountingPayload:
    def __init__(self, value: bytes) -> None:
        self.value = value
        self.copy_count = 0

    def __bytes__(self) -> bytes:
        self.copy_count += 1
        return self.value


def test_zenoh_latest_binary_stream_transport_keeps_latest_payload_only() -> None:
    transport = ZenohLatestBinaryStreamTransport(
        key_expr="f8/test/stream/raw",
        session=_Session(),
        log_context="test",
    )
    try:
        transport._on_sample(_Sample(b"old"))
        transport._on_sample(_Sample(b"new"))

        assert transport.poll_latest_raw() == b"new"
        assert transport.poll_latest_raw() is None
    finally:
        transport.close()


def test_zenoh_binary_stream_bounded_queue_preserves_order_and_drops_oldest() -> None:
    transport = ZenohLatestBinaryStreamTransport(
        key_expr="f8/test/stream/audio",
        session=_Session(),
        max_pending_samples=2,
    )
    try:
        for payload in (b"old", b"middle", b"new"):
            transport._on_sample(_Sample(payload))
        assert transport.poll_latest_raw() == b"middle"
        assert transport.poll_latest_raw() == b"new"
        assert transport.poll_latest_raw() is None
    finally:
        transport.close()


def test_zenoh_latest_binary_stream_transport_wait_returns_none_after_close() -> None:
    transport = ZenohLatestBinaryStreamTransport(
        key_expr="f8/test/stream/closed",
        session=_Session(),
        log_context="test",
    )
    transport.close()

    assert transport.wait_latest_raw(timeout_ms=10) is None


def test_zenoh_latest_binary_stream_transport_throttles_before_payload_copy() -> None:
    transport = ZenohLatestBinaryStreamTransport(
        key_expr="f8/test/stream/throttled",
        session=_Session(),
        log_context="test",
        min_sample_interval_ms=100,
    )
    try:
        accepted_payload = _CountingPayload(b"accepted")
        dropped_payload = _CountingPayload(b"dropped")

        transport._on_sample(_Sample(accepted_payload))
        transport._on_sample(_Sample(dropped_payload))

        assert accepted_payload.copy_count == 1
        assert dropped_payload.copy_count == 0
        assert transport.poll_latest_raw() == b"accepted"
        assert transport.poll_latest_raw() is None

        transport.set_min_sample_interval_ms(0)
        transport._on_sample(_Sample(dropped_payload))

        assert dropped_payload.copy_count == 1
        assert transport.poll_latest_raw() == b"dropped"
    finally:
        transport.close()
