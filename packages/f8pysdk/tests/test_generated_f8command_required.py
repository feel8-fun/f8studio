from f8pysdk.codec import validate_as
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.specs import F8Command  # noqa: E402


class F8CommandProtectionTests(unittest.TestCase):
    def test_definition_protected_defaults_to_false(self) -> None:
        cmd = F8Command(name="ping", params=[])
        self.assertFalse(cmd.definitionProtected)

    def test_definition_protected_parses_when_provided(self) -> None:
        cmd = validate_as(F8Command, {"name": "ping", "definitionProtected": True, "params": []})
        self.assertTrue(cmd.definitionProtected)


if __name__ == "__main__":
    unittest.main()
