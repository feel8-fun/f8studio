import os
import sys
import unittest
from unittest.mock import patch

from qtpy import QtWidgets


PKG_STUDIO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PKG_STUDIO not in sys.path:
    sys.path.insert(0, PKG_STUDIO)


try:
    import NodeGraphQt  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    NodeGraphQt = None  # type: ignore[assignment]


@unittest.skipIf(NodeGraphQt is None, "NodeGraphQt not installed in this environment")
class GraphCreateVariantNodeTests(unittest.TestCase):
    def test_create_variant_node_dispatches(self) -> None:
        from NodeGraphQt import BaseNode
        from f8pystudio.nodegraph.node_graph import F8StudioGraph
        from f8pystudio.variants.variant_ids import build_variant_node_type

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        _ = app

        class DummyNode(BaseNode):
            __identifier__ = "svc.test"
            NODE_NAME = "Dummy"

        graph = F8StudioGraph()
        graph.node_factory.clear_registered_nodes()
        graph.node_factory.register_node(DummyNode)
        registered_base_type = next(iter(graph.node_factory.nodes.keys()))

        variant_payload = {
            "variantId": "abc123",
            "baseNodeType": registered_base_type,
            "name": "Variant Name",
            "spec": {"schemaVersion": "f8operator/1", "serviceClass": "svc.test", "operatorClass": "f8.test"},
        }
        called: dict[str, str] = {}

        def _fake_apply_variant_to_node(self, *, node, variant_id, variant_name, variant_spec_json) -> None:
            _ = node
            _ = variant_spec_json
            called["variant_id"] = str(variant_id)
            called["variant_name"] = str(variant_name)

        with patch.object(F8StudioGraph, "_variant_record", return_value=variant_payload):
            with patch.object(F8StudioGraph, "_apply_variant_to_node", _fake_apply_variant_to_node):
                node = graph.create_node(build_variant_node_type("abc123"), push_undo=False)

        self.assertIsNotNone(node)
        self.assertEqual(called.get("variant_id"), "abc123")
        self.assertEqual(called.get("variant_name"), "Variant Name")


if __name__ == "__main__":
    unittest.main()
