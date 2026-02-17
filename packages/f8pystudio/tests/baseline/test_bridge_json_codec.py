from __future__ import annotations

import unittest

from _bootstrap import ensure_package_importable

ensure_package_importable()

from f8pystudio.bridge.json_codec import coerce_json_dict, coerce_json_value


class _FakeModelDump:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        if mode != "json":
            raise ValueError("mode must be json")
        return {"x": 1, "nested": {"ok": True}}


class _FakeRoot:
    root = {"field": "value"}


class BridgeJsonCodecTests(unittest.TestCase):
    def test_coerce_json_value_primitives(self) -> None:
        self.assertEqual(coerce_json_value("a"), "a")
        self.assertEqual(coerce_json_value(1), 1)
        self.assertEqual(coerce_json_value(True), True)

    def test_coerce_json_value_model_dump(self) -> None:
        payload = coerce_json_value(_FakeModelDump())
        self.assertEqual(payload, {"x": 1, "nested": {"ok": True}})

    def test_coerce_json_value_root(self) -> None:
        payload = coerce_json_value(_FakeRoot())
        self.assertEqual(payload, {"field": "value"})

    def test_coerce_json_dict_wraps_non_dict(self) -> None:
        payload = coerce_json_dict(["a", "b"])
        self.assertEqual(payload, {"value": ["a", "b"]})


if __name__ == "__main__":
    unittest.main()
