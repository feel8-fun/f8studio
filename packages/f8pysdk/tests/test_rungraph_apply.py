import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.generated import (  # noqa: E402
    F8RuntimeGraph,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
)
from f8pysdk.schema_helpers import string_schema  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402


class RungraphApplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_apply_rungraph_accepts_decoded_model(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svc")

        service_node = F8RuntimeNode(
            nodeId="svc",
            serviceId="svc",
            serviceClass="svc",
            operatorClass=None,
            stateFields=[
                F8StateSpec(name="svcId", valueSchema=string_schema(), access=F8StateAccess.ro),
            ],
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[service_node], edges=[])

        await bus.set_rungraph(graph)
        self.assertIsNotNone(bus._graph)


if __name__ == "__main__":
    unittest.main()
