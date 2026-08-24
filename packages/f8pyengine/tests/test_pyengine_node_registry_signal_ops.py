from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.specs import F8OperatorSpec, F8StateAccess, F8StateSpec  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.pyengine_node_registry import register_pyengine_specs  # noqa: E402


def _state_field(spec: F8OperatorSpec, name: str) -> F8StateSpec:
    matches = [field for field in list(spec.stateFields or []) if field.name == name]
    if len(matches) != 1:
        raise AssertionError(f"expected one state field named {name!r}, got {len(matches)}")
    return matches[0]


class PyEngineSignalOperatorRegistryTests(unittest.TestCase):
    def test_signal_processing_operators_are_registered(self) -> None:
        reg = create_runtime_node_registry()
        register_pyengine_specs(Registry.wrap(reg))
        desc = reg.describe(SERVICE_CLASS)
        operator_classes = {str(spec.operatorClass or "") for spec in list(desc.operators or [])}

        self.assertIn("f8.detrend", operator_classes)
        self.assertIn("f8.lowpass_filter", operator_classes)
        self.assertIn("f8.highpass_filter", operator_classes)
        self.assertIn("f8.bandpass_filter", operator_classes)
        self.assertIn("f8.periodicity_detector", operator_classes)
        self.assertIn("f8.udp_in", operator_classes)
        self.assertIn("f8.skeleton_decoder", operator_classes)
        self.assertIn("f8.vmc_decoder", operator_classes)
        self.assertIn("f8.contact_pose_axes", operator_classes)

    def test_operator_palette_categories_are_grouped_by_function(self) -> None:
        reg = create_runtime_node_registry()
        register_pyengine_specs(Registry.wrap(reg))
        desc = reg.describe(SERVICE_CLASS)
        operators = {str(spec.operatorClass or ""): spec for spec in list(desc.operators or [])}

        self.assertEqual(str(operators["f8.udp_in"].paletteCategory or ""), "f8.pyengine.input")
        self.assertEqual(str(operators["f8.udp_out"].paletteCategory or ""), "f8.pyengine.output")
        self.assertEqual(str(operators["f8.exec_sequence"].paletteCategory or ""), "f8.pyengine.execution")
        self.assertEqual(str(operators["f8.lowpass_filter"].paletteCategory or ""), "f8.pyengine.signal")
        self.assertEqual(str(operators["f8.python_script"].paletteCategory or ""), "f8.pyengine.expr")
        self.assertEqual(str(operators["f8.bone_filter"].paletteCategory or ""), "f8.pyengine.motion")
        self.assertEqual(str(operators["f8.skeleton_decoder"].paletteCategory or ""), "f8.pyengine.motion")
        self.assertEqual(str(operators["f8.vmc_decoder"].paletteCategory or ""), "f8.pyengine.motion")
        self.assertEqual(str(operators["f8.contact_pose_axes"].paletteCategory or ""), "f8.pyengine.motion")
        self.assertEqual(str(operators["f8.print"].paletteCategory or ""), "f8.pyengine.debug")

    def test_configuration_state_fields_are_read_write(self) -> None:
        reg = create_runtime_node_registry()
        register_pyengine_specs(Registry.wrap(reg))
        desc = reg.describe(SERVICE_CLASS)
        operators = {str(spec.operatorClass or ""): spec for spec in list(desc.operators or [])}

        self.assertEqual(_state_field(operators["f8.print"], "strip").access, F8StateAccess.rw)
        self.assertEqual(_state_field(operators["f8.serial_out"], "baudrate").access, F8StateAccess.rw)


if __name__ == "__main__":
    unittest.main()
