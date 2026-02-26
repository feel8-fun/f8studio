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
from f8pysdk.rungraph_validation import (  # noqa: E402
    validate_data_edges_or_raise,
    validate_exec_edges_or_raise,
    validate_state_edge_targets_writable_or_raise,
)
from f8pysdk.schema_helpers import string_schema  # noqa: E402


class RungraphValidationTests(unittest.TestCase):
    def test_rejects_cross_service_exec_edge(self) -> None:
        source = F8RuntimeNode(
            nodeId="opA",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass="OpA",
        )
        target = F8RuntimeNode(
            nodeId="opB",
            serviceId="svcB",
            serviceClass="svcB",
            operatorClass="OpB",
        )
        edge = F8Edge(
            edgeId="e1",
            fromServiceId="svcA",
            fromOperatorId="opA",
            fromPort="next",
            toServiceId="svcB",
            toOperatorId="opB",
            toPort="in",
            kind=F8EdgeKindEnum.exec,
            strategy=F8EdgeStrategyEnum.latest,
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[source, target], edges=[edge])
        with self.assertRaises(ValueError):
            validate_exec_edges_or_raise(graph)

    def test_rejects_exec_edge_with_non_operator_endpoint(self) -> None:
        source = F8RuntimeNode(
            nodeId="opA",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass="OpA",
        )
        service_node = F8RuntimeNode(
            nodeId="svcNode",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass=None,
        )
        edge = F8Edge(
            edgeId="e1",
            fromServiceId="svcA",
            fromOperatorId="opA",
            fromPort="next",
            toServiceId="svcA",
            toOperatorId="svcNode",
            toPort="in",
            kind=F8EdgeKindEnum.exec,
            strategy=F8EdgeStrategyEnum.latest,
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[source, service_node], edges=[edge])
        with self.assertRaises(ValueError):
            validate_exec_edges_or_raise(graph)

    def test_rejects_exec_single_out_violation(self) -> None:
        a = F8RuntimeNode(nodeId="opA", serviceId="svcA", serviceClass="svcA", operatorClass="OpA")
        b = F8RuntimeNode(nodeId="opB", serviceId="svcA", serviceClass="svcA", operatorClass="OpB")
        c = F8RuntimeNode(nodeId="opC", serviceId="svcA", serviceClass="svcA", operatorClass="OpC")
        edges = [
            F8Edge(
                edgeId="e1",
                fromServiceId="svcA",
                fromOperatorId="opA",
                fromPort="next",
                toServiceId="svcA",
                toOperatorId="opB",
                toPort="in",
                kind=F8EdgeKindEnum.exec,
                strategy=F8EdgeStrategyEnum.latest,
            ),
            F8Edge(
                edgeId="e2",
                fromServiceId="svcA",
                fromOperatorId="opA",
                fromPort="next",
                toServiceId="svcA",
                toOperatorId="opC",
                toPort="in",
                kind=F8EdgeKindEnum.exec,
                strategy=F8EdgeStrategyEnum.latest,
            ),
        ]
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[a, b, c], edges=edges)
        with self.assertRaises(ValueError):
            validate_exec_edges_or_raise(graph)

    def test_rejects_exec_single_in_violation(self) -> None:
        a = F8RuntimeNode(nodeId="opA", serviceId="svcA", serviceClass="svcA", operatorClass="OpA")
        b = F8RuntimeNode(nodeId="opB", serviceId="svcA", serviceClass="svcA", operatorClass="OpB")
        c = F8RuntimeNode(nodeId="opC", serviceId="svcA", serviceClass="svcA", operatorClass="OpC")
        edges = [
            F8Edge(
                edgeId="e1",
                fromServiceId="svcA",
                fromOperatorId="opA",
                fromPort="next",
                toServiceId="svcA",
                toOperatorId="opC",
                toPort="in",
                kind=F8EdgeKindEnum.exec,
                strategy=F8EdgeStrategyEnum.latest,
            ),
            F8Edge(
                edgeId="e2",
                fromServiceId="svcA",
                fromOperatorId="opB",
                fromPort="next",
                toServiceId="svcA",
                toOperatorId="opC",
                toPort="in",
                kind=F8EdgeKindEnum.exec,
                strategy=F8EdgeStrategyEnum.latest,
            ),
        ]
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[a, b, c], edges=edges)
        with self.assertRaises(ValueError):
            validate_exec_edges_or_raise(graph)

    def test_rejects_data_single_input_violation(self) -> None:
        a = F8RuntimeNode(nodeId="opA", serviceId="svcA", serviceClass="svcA", operatorClass="OpA")
        b = F8RuntimeNode(nodeId="opB", serviceId="svcA", serviceClass="svcA", operatorClass="OpB")
        c = F8RuntimeNode(nodeId="opC", serviceId="svcA", serviceClass="svcA", operatorClass="OpC")
        edges = [
            F8Edge(
                edgeId="e1",
                fromServiceId="svcA",
                fromOperatorId="opA",
                fromPort="out1",
                toServiceId="svcA",
                toOperatorId="opC",
                toPort="in1",
                kind=F8EdgeKindEnum.data,
                strategy=F8EdgeStrategyEnum.latest,
            ),
            F8Edge(
                edgeId="e2",
                fromServiceId="svcA",
                fromOperatorId="opB",
                fromPort="out2",
                toServiceId="svcA",
                toOperatorId="opC",
                toPort="in1",
                kind=F8EdgeKindEnum.data,
                strategy=F8EdgeStrategyEnum.latest,
            ),
        ]
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[a, b, c], edges=edges)
        with self.assertRaises(ValueError):
            validate_data_edges_or_raise(graph)

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

    def test_local_service_cross_service_source_missing_is_allowed(self) -> None:
        # Simulate a per-service half-graph for svcB: inbound edge is present,
        # but upstream node/state fields from svcA are intentionally absent.
        target = F8RuntimeNode(
            nodeId="opB",
            serviceId="svcB",
            serviceClass="svcB",
            operatorClass="OpB",
            stateFields=[F8StateSpec(name="input", valueSchema=string_schema(), access=F8StateAccess.rw)],
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
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[target], edges=[edge])
        validate_state_edge_targets_writable_or_raise(graph, local_service_id="svcB")


if __name__ == "__main__":
    unittest.main()
