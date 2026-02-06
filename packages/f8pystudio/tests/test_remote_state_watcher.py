import json
import os
import sys
import unittest


PKG_STUDIO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
for p in (PKG_STUDIO, PKG_SDK):
    if p not in sys.path:
        sys.path.insert(0, p)


from f8pystudio.remote_state_watcher import RemoteStateWatcher  # noqa: E402


class RemoteStateWatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_allows_value_change_with_older_ts(self) -> None:
        events: list[tuple[str, str, str, object, int]] = []

        async def _on_state(service_id: str, node_id: str, field: str, value: object, ts_ms: int, meta: dict) -> None:
            _ = meta
            events.append((service_id, node_id, field, value, int(ts_ms)))

        w = RemoteStateWatcher(
            nats_url="nats://127.0.0.1:4222",
            studio_service_id="studio",
            on_state=_on_state,
        )

        key = "nodes.n1.state.status"
        await w._on_kv("svc", key, json.dumps({"value": "a", "ts_ms": 2000}).encode("utf-8"))
        await w._on_kv("svc", key, json.dumps({"value": "b", "ts_ms": 3000}).encode("utf-8"))
        # Simulate a legitimate "revert" where the upstream timestamp is older but the value changes.
        await w._on_kv("svc", key, json.dumps({"value": "a", "ts_ms": 2000}).encode("utf-8"))

        self.assertEqual([e[3] for e in events], ["a", "b", "a"])

    async def test_dedupes_repeated_value_even_with_newer_ts(self) -> None:
        events: list[tuple[str, str, str, object, int]] = []

        async def _on_state(service_id: str, node_id: str, field: str, value: object, ts_ms: int, meta: dict) -> None:
            _ = (service_id, node_id, field, ts_ms, meta)
            events.append(("svc", "n1", "status", value, int(ts_ms)))

        w = RemoteStateWatcher(
            nats_url="nats://127.0.0.1:4222",
            studio_service_id="studio",
            on_state=_on_state,
        )

        key = "nodes.n1.state.status"
        await w._on_kv("svc", key, json.dumps({"value": {"x": 1}, "ts_ms": 10}).encode("utf-8"))
        await w._on_kv("svc", key, json.dumps({"value": {"x": 1}, "ts_ms": 20}).encode("utf-8"))

        self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
