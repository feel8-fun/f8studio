import os
import sys
import unittest


PKG_STUDIO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PKG_STUDIO not in sys.path:
    sys.path.insert(0, PKG_STUDIO)


class VariantNodeTypeParseTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        from f8pystudio.variants.variant_ids import build_variant_node_type, is_variant_node_type, parse_variant_node_type

        node_type = build_variant_node_type("abc123")
        self.assertTrue(is_variant_node_type(node_type))
        self.assertEqual(parse_variant_node_type(node_type), "abc123")

    def test_legacy_prefix(self) -> None:
        from f8pystudio.variants.variant_ids import parse_variant_node_type

        self.assertEqual(parse_variant_node_type("__variant__:hello-world"), "hello_world")


if __name__ == "__main__":
    unittest.main()
