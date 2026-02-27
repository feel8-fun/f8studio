import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.service_bus.bus import ServiceBus, ServiceBusConfig  # noqa: E402
from f8pysdk.service_bus.routing_data import buffer_input  # noqa: E402
from f8pysdk.runtime_node import RuntimeNode  # noqa: E402


class ServiceBusCapTests(unittest.TestCase):
    def test_state_cache_lru_cap(self) -> None:
        bus = ServiceBus(ServiceBusConfig(service_id="svc", state_cache_max_entries=2))

        bus._state_cache[("n", "a")] = ("va", 1)
        bus._state_cache[("n", "b")] = ("vb", 2)
        _ = bus._state_cache.get(("n", "a"))
        bus._state_cache[("n", "c")] = ("vc", 3)

        self.assertEqual(len(bus._state_cache), 2)
        self.assertIn(("n", "a"), bus._state_cache)
        self.assertIn(("n", "c"), bus._state_cache)
        self.assertNotIn(("n", "b"), bus._state_cache)

    def test_data_input_buffer_lru_cap(self) -> None:
        bus = ServiceBus(ServiceBusConfig(service_id="svc", data_input_max_buffers=2))

        buffer_input(bus, "n1", "in", 1, ts_ms=1, edge=None, ctx_id=None)
        buffer_input(bus, "n2", "in", 2, ts_ms=2, edge=None, ctx_id=None)
        _ = bus._data_inputs.get(("n1", "in"))
        buffer_input(bus, "n3", "in", 3, ts_ms=3, edge=None, ctx_id=None)

        self.assertEqual(len(bus._data_inputs), 2)
        self.assertIn(("n1", "in"), bus._data_inputs)
        self.assertIn(("n3", "in"), bus._data_inputs)
        self.assertNotIn(("n2", "in"), bus._data_inputs)

    def test_data_input_default_queue_size(self) -> None:
        bus = ServiceBus(ServiceBusConfig(service_id="svc", data_input_default_queue_size=2))

        buffer_input(bus, "n1", "in", "v1", ts_ms=1, edge=None, ctx_id=None)
        buffer_input(bus, "n1", "in", "v2", ts_ms=2, edge=None, ctx_id=None)
        buffer_input(bus, "n1", "in", "v3", ts_ms=3, edge=None, ctx_id=None)

        buf = bus._data_inputs[("n1", "in")]
        self.assertEqual(list(buf.queue), [("v2", 2), ("v3", 3)])

    def test_get_state_cached_hit_and_miss(self) -> None:
        bus = ServiceBus(ServiceBusConfig(service_id="svc"))
        self.assertEqual(bus.get_state_cached("n1", "a", 123), 123)
        bus._state_cache[("n1", "a")] = ("valueA", 10)
        self.assertEqual(bus.get_state_cached("n1", "a", None), "valueA")

    def test_runtime_node_get_state_cached(self) -> None:
        node = RuntimeNode(node_id="n1")
        self.assertEqual(node.get_state_cached("k", "d"), "d")

        bus = ServiceBus(ServiceBusConfig(service_id="svc"))
        bus._state_cache[("n1", "k")] = ("vk", 11)
        node.attach(bus)
        self.assertEqual(node.get_state_cached("k", "d"), "vk")


if __name__ == "__main__":
    unittest.main()
