import os
import sys
import unittest
from types import SimpleNamespace


PKG_STUDIO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PKG_STUDIO not in sys.path:
    sys.path.insert(0, PKG_STUDIO)


class VariantComposeTests(unittest.TestCase):
    def test_compose_applies_ui_and_defaults(self) -> None:
        from f8pysdk import (
            F8DataPortSpec,
            F8OperatorSchemaVersion,
            F8OperatorSpec,
            F8StateAccess,
            F8StateSpec,
            any_schema,
            string_schema,
        )
        from f8pystudio.variants.variant_compose import build_variant_record_from_node

        spec = F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass="svc.test",
            operatorClass="f8.test",
            version="0.0.1",
            label="Base Label",
            description="Base Desc",
            tags=["base"],
            execInPorts=["exec"],
            execOutPorts=["exec"],
            editableExecInPorts=True,
            editableExecOutPorts=True,
            dataInPorts=[F8DataPortSpec(name="msg", valueSchema=any_schema(), showOnNode=True)],
            dataOutPorts=[F8DataPortSpec(name="out", valueSchema=any_schema(), showOnNode=True)],
            editableDataInPorts=True,
            editableDataOutPorts=True,
            stateFields=[
                F8StateSpec(
                    name="code",
                    valueSchema=string_schema(default="old"),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                    uiControl="code",
                )
            ],
            editableStateFields=True,
        )

        node = SimpleNamespace(
            spec=spec,
            type_="svc.test.f8.test",
            model=SimpleNamespace(custom_properties={"code": "print('hello')"}),
            ui_overrides=lambda: {
                "stateFields": {"code": {"showOnNode": True, "label": "Code Label"}},
                "dataPorts": {"in": {"msg": {"showOnNode": False}}},
            },
        )

        rec = build_variant_record_from_node(
            node=node,
            name="My Variant",
            description="My Desc",
            tags=["x"],
        )
        self.assertEqual(rec.serviceClass, "svc.test")
        self.assertEqual(rec.operatorClass, "f8.test")
        self.assertEqual(rec.spec["label"], "My Variant")
        self.assertEqual(rec.spec["description"], "My Desc")
        self.assertEqual(rec.spec["tags"], ["x"])
        code_field = rec.spec["stateFields"][0]
        self.assertTrue(code_field["showOnNode"])
        self.assertEqual(code_field["label"], "Code Label")
        self.assertEqual(code_field["valueSchema"]["default"], "print('hello')")
        self.assertFalse(rec.spec["dataInPorts"][0]["showOnNode"])


if __name__ == "__main__":
    unittest.main()
