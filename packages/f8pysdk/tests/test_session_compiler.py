import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.generated import (  # noqa: E402
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8ServiceSchemaVersion,
    F8ServiceSpec,
)
from f8pysdk.service_runtime_tools.catalog import ServiceCatalog  # noqa: E402
from f8pysdk.service_runtime_tools.session_compiler import (  # noqa: E402
    compile_runtime_graphs_from_session_layout,
)


class SessionCompilerTests(unittest.TestCase):
    @staticmethod
    def _service_spec(service_class: str, *, label: str = "PyEngine") -> F8ServiceSpec:
        return F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=service_class,
            version="0.0.1",
            label=label,
        )

    @staticmethod
    def _operator_spec(service_class: str, operator_class: str, *, label: str = "Op") -> F8OperatorSpec:
        return F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass=service_class,
            operatorClass=operator_class,
            version="0.0.1",
            label=label,
            execOutPorts=["next"],
            execInPorts=["in"],
        )

    def setUp(self) -> None:
        self.catalog = ServiceCatalog.instance()
        self.catalog.clear()
        self.catalog.register_service(self._service_spec("f8.pyengine"))
        self.catalog.register_operator(self._operator_spec("f8.pyengine", "f8.pyengine.op"))

    def test_skip_pystudio_nodes_without_error(self) -> None:
        layout = {
            "nodes": {
                "svc1": {
                    "id": "svc1",
                    "f8_spec": self._service_spec("f8.pyengine").model_dump(mode="json"),
                },
                "op1": {
                    "id": "op1",
                    "f8_spec": self._operator_spec("f8.pyengine", "f8.pyengine.op").model_dump(mode="json"),
                    "custom": {"svcId": "svc1"},
                },
                "studio_op": {
                    "id": "studio_op",
                    "f8_spec": F8OperatorSpec(
                        schemaVersion=F8OperatorSchemaVersion.f8operator_1,
                        serviceClass="f8.pystudio",
                        operatorClass="f8.pystudio.viz",
                        version="0.0.1",
                        label="Studio Viz",
                        execInPorts=["in"],
                    ).model_dump(mode="json"),
                    "custom": {"svcId": "studio"},
                },
            },
            "connections": [
                {"out": ["op1", "next[E]"], "in": ["studio_op", "[E]in"]},
            ],
        }

        compiled = compile_runtime_graphs_from_session_layout(layout=layout, catalog=self.catalog)
        service_classes = {str(s.serviceClass) for s in compiled.global_graph.services}
        operator_classes = {str(n.operatorClass or "") for n in compiled.global_graph.nodes}
        self.assertIn("f8.pyengine", service_classes)
        self.assertNotIn("f8.pystudio", service_classes)
        self.assertNotIn("f8.pystudio.viz", operator_classes)

    def test_unknown_service_raises(self) -> None:
        layout = {
            "nodes": {
                "svc_unknown": {
                    "id": "svc_unknown",
                    "f8_spec": self._service_spec("f8.unknown", label="Unknown").model_dump(mode="json"),
                }
            },
            "connections": [],
        }
        with self.assertRaises(ValueError):
            compile_runtime_graphs_from_session_layout(layout=layout, catalog=self.catalog)

    def test_unknown_operator_raises(self) -> None:
        layout = {
            "nodes": {
                "svc1": {
                    "id": "svc1",
                    "f8_spec": self._service_spec("f8.pyengine").model_dump(mode="json"),
                },
                "op_unknown": {
                    "id": "op_unknown",
                    "f8_spec": F8OperatorSpec(
                        schemaVersion=F8OperatorSchemaVersion.f8operator_1,
                        serviceClass="f8.pyengine",
                        operatorClass="f8.pyengine.unknown",
                        version="0.0.1",
                        label="Unknown Operator",
                    ).model_dump(mode="json"),
                    "custom": {"svcId": "svc1"},
                },
            },
            "connections": [],
        }
        with self.assertRaises(ValueError):
            compile_runtime_graphs_from_session_layout(layout=layout, catalog=self.catalog)


if __name__ == "__main__":
    unittest.main()
