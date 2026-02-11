import os
import sys
import unittest


PKG_STUDIO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PKG_STUDIO not in sys.path:
    sys.path.insert(0, PKG_STUDIO)


try:
    import NodeGraphQt  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    NodeGraphQt = None  # type: ignore[assignment]


@unittest.skipIf(NodeGraphQt is None, "NodeGraphQt not installed in this environment")
class NodeModelSessionCompatTests(unittest.TestCase):
    def test_ignores_unknown_property(self) -> None:
        from f8pystudio.nodegraph.node_model import F8StudioNodeModel  # noqa: E402

        m = F8StudioNodeModel()
        # Should not raise NodePropertyError.
        m.set_property("Event", {"x": 1})


if __name__ == "__main__":
    unittest.main()
