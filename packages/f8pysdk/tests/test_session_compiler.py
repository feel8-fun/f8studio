from f8pysdk.specs import exec_port_specs
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.codec import dump_json
from f8pysdk.command import command_input_state_field, command_output_state_field
from f8pysdk.specs import (  # noqa: E402
    F8Command,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8ServiceSchemaVersion,
    F8ServiceSpec,
)
from f8pysdk.service_runtime_tools.inventory.catalog import ServiceCatalog  # noqa: E402
from f8pysdk.service_runtime_tools.session.compiler import (  # noqa: E402
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
            execOutPorts=exec_port_specs(["next"]),
            execInPorts=exec_port_specs(["in"]),
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
                    "f8_spec": dump_json(self._service_spec("f8.pyengine"), mode="json"),
                },
                "op1": {
                    "id": "op1",
                    "f8_spec": dump_json(self._operator_spec("f8.pyengine", "f8.pyengine.op"), mode="json"),
                    "custom": {"svcId": "svc1"},
                },
                "studio_op": {
                    "id": "studio_op",
                    "f8_spec": dump_json(
                        F8OperatorSpec(
                            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
                            serviceClass="f8.pystudio",
                            operatorClass="f8.pystudio.viz",
                            version="0.0.1",
                            label="Studio Viz",
                            execInPorts=exec_port_specs(["in"]),
                        ),
                        mode="json",
                    ),
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
                    "f8_spec": dump_json(self._service_spec("f8.unknown", label="Unknown"), mode="json"),
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
                    "f8_spec": dump_json(self._service_spec("f8.pyengine"), mode="json"),
                },
                "op_unknown": {
                    "id": "op_unknown",
                    "f8_spec": dump_json(
                        F8OperatorSpec(
                            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
                            serviceClass="f8.pyengine",
                            operatorClass="f8.pyengine.unknown",
                            version="0.0.1",
                            label="Unknown Operator",
                        ),
                        mode="json",
                    ),
                    "custom": {"svcId": "svc1"},
                },
            },
            "connections": [],
        }
        with self.assertRaises(ValueError):
            compile_runtime_graphs_from_session_layout(layout=layout, catalog=self.catalog)

    def test_invalid_cross_service_exec_connection_raises(self) -> None:
        layout = {
            "nodes": {
                "svc1": {
                    "id": "svc1",
                    "f8_spec": dump_json(self._service_spec("f8.pyengine"), mode="json"),
                },
                "svc2": {
                    "id": "svc2",
                    "f8_spec": dump_json(self._service_spec("f8.pyengine"), mode="json"),
                },
                "op1": {
                    "id": "op1",
                    "f8_spec": dump_json(self._operator_spec("f8.pyengine", "f8.pyengine.op"), mode="json"),
                    "custom": {"svcId": "svc1"},
                },
                "op2": {
                    "id": "op2",
                    "f8_spec": dump_json(self._operator_spec("f8.pyengine", "f8.pyengine.op"), mode="json"),
                    "custom": {"svcId": "svc2"},
                },
            },
            "connections": [
                {"out": ["op1", "next[E]"], "in": ["op2", "[E]in"]},
            ],
        }
        with self.assertRaises(ValueError):
            compile_runtime_graphs_from_session_layout(layout=layout, catalog=self.catalog)

    def test_missing_locked_node_raises(self) -> None:
        layout = {
            "nodes": {
                "svc_missing": {
                    "id": "svc_missing",
                    "f8_spec": dump_json(self._service_spec("f8.pyengine"), mode="json"),
                    "f8_sys": {
                        "missingLocked": True,
                        "missingType": "svc.some.missing",
                    },
                }
            },
            "connections": [],
        }
        with self.assertRaises(ValueError) as ctx:
            compile_runtime_graphs_from_session_layout(layout=layout, catalog=self.catalog)
        self.assertIn("missing dependency node", str(ctx.exception))
        self.assertIn("svc_missing", str(ctx.exception))

    def test_compiled_graph_omits_unset_optional_fields_in_payload(self) -> None:
        service_spec = F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass="f8.pyengine",
            version="0.0.1",
            label="",
        )
        operator_spec = F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass="f8.pyengine",
            operatorClass="f8.pyengine.op",
            version="0.0.1",
            label="Op",
            execInPorts=exec_port_specs(["in"]),
            execOutPorts=exec_port_specs(["next"]),
        )
        layout = {
            "nodes": {
                "svc1": {
                    "id": "svc1",
                    "f8_spec": dump_json(service_spec, mode="json"),
                },
                "op1": {
                    "id": "op1",
                    "f8_spec": dump_json(operator_spec, mode="json"),
                    "custom": {"svcId": "svc1"},
                },
            },
            "connections": [],
        }

        compiled = compile_runtime_graphs_from_session_layout(layout=layout, catalog=self.catalog)
        payload = dump_json(compiled.global_graph, mode="json", by_alias=True)
        self.assertIsInstance(payload, dict)

        services_payload = payload.get("services")
        self.assertIsInstance(services_payload, list)
        self.assertGreaterEqual(len(services_payload), 1)
        self.assertNotIn("label", services_payload[0])

        nodes_payload = payload.get("nodes")
        self.assertIsInstance(nodes_payload, list)
        self.assertGreaterEqual(len(nodes_payload), 2)
        service_node_payload = next(item for item in nodes_payload if item.get("nodeId") == "svc1")
        operator_node_payload = next(item for item in nodes_payload if item.get("nodeId") == "op1")
        self.assertNotIn("operatorClass", service_node_payload)
        self.assertNotIn("stateValues", service_node_payload)
        self.assertNotIn("stateValues", operator_node_payload)

    def test_operator_commands_compile_hidden_backing_states(self) -> None:
        operator_spec = F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass="f8.pyengine",
            operatorClass="f8.pyengine.op",
            version="0.0.1",
            label="Op",
            commands=[F8Command(name="run", params=[])],
        )
        layout = {
            "nodes": {
                "svc1": {
                    "id": "svc1",
                    "f8_spec": dump_json(self._service_spec("f8.pyengine"), mode="json"),
                },
                "op1": {
                    "id": "op1",
                    "f8_spec": dump_json(operator_spec, mode="json"),
                    "custom": {"svcId": "svc1"},
                },
            },
            "connections": [],
        }

        compiled = compile_runtime_graphs_from_session_layout(layout=layout, catalog=self.catalog)
        operator_node = next(node for node in list(compiled.global_graph.nodes or []) if str(node.nodeId) == "op1")
        field_names = {str(field.name or "") for field in list(operator_node.stateFields or [])}

        self.assertIn(command_input_state_field("run"), field_names)
        self.assertIn(command_output_state_field("run"), field_names)

    def test_compiler_ignores_editor_only_f8_layers_metadata(self) -> None:
        layout = {
            "f8_layers": [
                {
                    "id": "base",
                    "label": "Base",
                    "defaultVisible": True,
                    "isBase": True,
                },
                {
                    "id": "logic",
                    "label": "Logic",
                    "defaultVisible": False,
                    "isBase": False,
                },
            ],
            "nodes": {
                "svc1": {
                    "id": "svc1",
                    "f8_spec": dump_json(self._service_spec("f8.pyengine"), mode="json"),
                    "f8_ui_state": {"layerIds": ["logic"]},
                },
                "op1": {
                    "id": "op1",
                    "f8_spec": dump_json(self._operator_spec("f8.pyengine", "f8.pyengine.op"), mode="json"),
                    "custom": {"svcId": "svc1"},
                    "f8_ui_state": {"layerIds": ["logic"]},
                },
            },
            "connections": [],
        }

        compiled = compile_runtime_graphs_from_session_layout(layout=layout, catalog=self.catalog)

        self.assertEqual(len(list(compiled.global_graph.nodes or [])), 2)
        self.assertEqual(len(list(compiled.global_graph.edges or [])), 0)


if __name__ == "__main__":
    unittest.main()
