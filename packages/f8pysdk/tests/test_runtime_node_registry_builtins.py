import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.specs import F8OperatorSpec, F8ServiceSpec, F8StateAccess, F8StateSpec  # noqa: E402
from f8pysdk.registry import create_runtime_node_registry  # noqa: E402
from f8pysdk.specs import boolean_schema, string_schema  # noqa: E402


class RuntimeNodeRegistryBuiltinTests(unittest.TestCase):
    def test_describe_force_overrides_builtin_fields(self) -> None:
        registry = create_runtime_node_registry()
        service_spec = F8ServiceSpec(
            serviceClass="f8.tests.svc",
            version="0.0.1",
            label="svc",
            stateFields=[
                F8StateSpec(name="active", valueSchema=boolean_schema(default=False), access=F8StateAccess.ro),
                F8StateSpec(name="svcId", valueSchema=string_schema(), access=F8StateAccess.rw),
                F8StateSpec(name="custom", valueSchema=string_schema(), access=F8StateAccess.rw),
            ],
        )
        operator_spec = F8OperatorSpec(
            serviceClass="f8.tests.svc",
            operatorClass="f8.tests.op",
            version="0.0.1",
            label="op",
            stateFields=[
                F8StateSpec(name="svcId", valueSchema=string_schema(), access=F8StateAccess.rw),
                F8StateSpec(name="operatorId", valueSchema=string_schema(), access=F8StateAccess.rw),
                F8StateSpec(name="threshold", valueSchema=string_schema(), access=F8StateAccess.rw),
            ],
        )
        registry.register_service_spec(service_spec)
        registry.register_operator_spec(operator_spec)

        desc = registry.describe("f8.tests.svc")
        service_fields = list(desc.service.stateFields or [])
        operator_fields = list(desc.operators[0].stateFields or [])

        self.assertEqual([str(x.name) for x in service_fields], ["custom", "active", "svcId"])
        self.assertEqual([str(x.name) for x in operator_fields], ["threshold", "svcId", "operatorId"])
        self.assertEqual(service_fields[-2].access, F8StateAccess.rw)
        self.assertTrue(bool(service_fields[-2].valueRequired))
        self.assertFalse(bool(service_fields[-2].showOnNode))
        self.assertEqual(service_fields[-1].access, F8StateAccess.ro)
        self.assertTrue(bool(service_fields[-1].valueRequired))
        self.assertTrue(bool(operator_fields[-2].valueRequired))
        self.assertTrue(bool(operator_fields[-1].valueRequired))
        service_out_ports = list(desc.service.dataOutPorts or [])
        monitor_ports = [port for port in service_out_ports if str(port.name or "") == "monitor"]
        self.assertEqual(len(monitor_ports), 1)
        self.assertTrue(bool(monitor_ports[0].definitionProtected))


if __name__ == "__main__":
    unittest.main()
