from __future__ import annotations

import json
import os
import re
import unittest
from pathlib import Path


# Decision and FBX playback currently have no C++ implementation/spec; Python
# script has the distinct cpython_script counterpart. This is an explicit
# coverage policy, not an assertion that pending C++ specs are executable.
PYTHON_ONLY_OPERATORS = {"f8.python_script", "f8.decision", "f8.fbx_skeleton_player"}


def cpp_describe() -> dict:
    root = os.environ.get("F8_CPP_DESCRIBE_ROOT")
    if not root:
        raise unittest.SkipTest("built C++ contract test requires F8_CPP_DESCRIBE_ROOT")
    return json.loads((Path(root) / "f8/cppengine/describe.json").read_text())


class CppEngineOperatorCoverageTest(unittest.TestCase):
    def test_cpp_sources_declare_python_operator_contracts(self) -> None:
        from f8pyengine.pyengine_service import build_app

        python_specs = {op["operatorClass"] for op in build_app().describe_json()["operators"]}
        sources = Path("extensions/f8cppengine/src")
        declared = set()
        for path in sources.glob("*.cpp"):
            declared.update(re.findall(r'"(f8\.[a-z_]+)"', path.read_text()))
        self.assertEqual(python_specs - declared, PYTHON_ONLY_OPERATORS)

    def test_cppengine_describes_all_pyengine_operators_except_python_script(self) -> None:
        from f8pyengine.pyengine_service import build_app

        pyengine = build_app().describe_json()
        cppengine = cpp_describe()

        py_operator_classes = {str(op["operatorClass"]) for op in pyengine["operators"]}
        cpp_operator_classes = {str(op["operatorClass"]) for op in cppengine["operators"]}

        expected_missing = PYTHON_ONLY_OPERATORS
        self.assertEqual(py_operator_classes - cpp_operator_classes, expected_missing)
        self.assertIn("f8.data_pick", cpp_operator_classes)
        self.assertIn("f8.lua_script", cpp_operator_classes)
        self.assertNotIn("f8.angelscript", cpp_operator_classes)

    def test_cppengine_data_pick_operator_shape(self) -> None:
        cppengine = cpp_describe()
        specs = {str(op["operatorClass"]): op for op in cppengine["operators"]}

        data_pick = specs["f8.data_pick"]
        self.assertEqual(data_pick["label"], "Data Pick")
        self.assertEqual(data_pick["paletteCategory"], "f8.cppengine.expr")
        self.assertEqual([port["name"] for port in data_pick["dataInPorts"]], ["msg"])
        self.assertEqual([port["name"] for port in data_pick["dataOutPorts"]], ["out"])

        state_fields = {str(field["name"]): field for field in data_pick["stateFields"]}
        self.assertEqual(state_fields["path"]["control"], {"kind": "textarea"})
        self.assertEqual(state_fields["path"]["valueSchema"]["default"], "")
        self.assertEqual(state_fields["valueType"]["valueSchema"]["enum"], ["any", "number", "string", "bool"])
        self.assertEqual(state_fields["fallback"]["control"], {"kind": "textarea", "language": "json"})
        self.assertIsNone(state_fields["fallback"]["valueSchema"]["default"])

    def test_cppengine_script_operators_ship_starter_templates(self) -> None:
        cppengine = cpp_describe()
        specs = {str(op["operatorClass"]): op for op in cppengine["operators"]}

        lua_code_field = specs["f8.lua_script"]["stateFields"][0]
        lua_code = lua_code_field["valueSchema"]["default"]
        self.assertEqual(lua_code_field["control"], {"kind": "code", "language": "lua"})
        self.assertEqual(lua_code_field["editorAssist"], {"version": 1, "language": "lua"})
        self.assertIn("on_exec(ctx, exec_in, inputs)", lua_code)
        self.assertIn("ctx:pull(\"msg\")", lua_code)
        self.assertIn("on_pull(ctx, port, inputs)", lua_code)


if __name__ == "__main__":
    unittest.main()
