import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk import F8Command  # noqa: E402


class F8CommandRequiredFieldTests(unittest.TestCase):
    def test_required_defaults_to_false(self) -> None:
        cmd = F8Command(name="ping", params=[])
        self.assertFalse(cmd.required)

    def test_required_parses_when_provided(self) -> None:
        cmd = F8Command.model_validate({"name": "ping", "required": True, "params": []})
        self.assertTrue(cmd.required)


if __name__ == "__main__":
    unittest.main()
