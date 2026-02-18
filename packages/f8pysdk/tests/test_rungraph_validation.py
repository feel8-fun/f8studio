import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.generated import (  # noqa: E402
    F8Edge,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8RuntimeGraph,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
)
from f8pysdk.rungraph_validation import validate_state_edge_targets_writable_or_raise  # noqa: E402
from f8pysdk.schema_helpers import string_schema  # noqa: E402


class RungraphValidationTests(unittest.TestCase):
    def test_rejects_state_edge_to_readonly_target(self) -> None:
        source = F8RuntimeNode(
            nodeId="opA",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass="OpA",
            stateFields=[F8StateSpec(name="out", valueSchema=string_schema(), access=F8StateAccess.rw)],
        )
        target = F8RuntimeNode(
            nodeId="opB",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass="OpB",
            stateFields=[F8StateSpec(name="input", valueSchema=string_schema(), access=F8StateAccess.ro)],
        )
        edge = F8Edge(
            edgeId="e1",
            fromServiceId="svcA",
            fromOperatorId="opA",
            fromPort="out",
            toServiceId="svcA",
            toOperatorId="opB",
            toPort="input",
            kind=F8EdgeKindEnum.state,
            strategy=F8EdgeStrategyEnum.latest,
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[source, target], edges=[edge])
        with self.assertRaises(ValueError):
            validate_state_edge_targets_writable_or_raise(graph)

    def test_local_service_filter_skips_other_services(self) -> None:
        source = F8RuntimeNode(
            nodeId="opA",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass="OpA",
            stateFields=[F8StateSpec(name="out", valueSchema=string_schema(), access=F8StateAccess.rw)],
        )
        target = F8RuntimeNode(
            nodeId="opB",
            serviceId="svcB",
            serviceClass="svcB",
            operatorClass="OpB",
            stateFields=[F8StateSpec(name="input", valueSchema=string_schema(), access=F8StateAccess.ro)],
        )
        edge = F8Edge(
            edgeId="e1",
            fromServiceId="svcA",
            fromOperatorId="opA",
            fromPort="out",
            toServiceId="svcB",
            toOperatorId="opB",
            toPort="input",
            kind=F8EdgeKindEnum.state,
            strategy=F8EdgeStrategyEnum.latest,
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[source, target], edges=[edge])
        validate_state_edge_targets_writable_or_raise(graph, local_service_id="svcA")


if __name__ == "__main__":
    unittest.main()
