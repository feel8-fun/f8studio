import os
import sys
import unittest


PKG_STUDIO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
for p in (PKG_STUDIO, PKG_SDK):
    if p not in sys.path:
        sys.path.insert(0, p)


from f8pysdk.runtime_node_registry import RuntimeNodeRegistry  # noqa: E402
from f8pystudio.constants import SERVICE_CLASS  # noqa: E402
from f8pystudio.operators import register_operator as register_studio_operators  # noqa: E402

try:
    from f8pystudio.render_nodes.registry import RenderNodeRegistry  # noqa: E402

    _HAS_RENDER_REGISTRY = True
except Exception:
    RenderNodeRegistry = None  # type: ignore[assignment]
    _HAS_RENDER_REGISTRY = False


class TCodeViewerRegistrationTests(unittest.TestCase):
    def test_operator_spec_registered(self) -> None:
        reg = RuntimeNodeRegistry()
        register_studio_operators(reg)
        specs = reg.operator_specs(SERVICE_CLASS)
        target = [s for s in specs if str(s.operatorClass or "") == "f8.tcode_viewer"]
        self.assertEqual(len(target), 1)
        self.assertEqual(str(target[0].rendererClass or ""), "pystudio_tcode_viewer")

    @unittest.skipUnless(_HAS_RENDER_REGISTRY, "render registry deps unavailable")
    def test_renderer_registered(self) -> None:
        assert RenderNodeRegistry is not None
        render_reg = RenderNodeRegistry()
        node_cls = render_reg.get("pystudio_tcode_viewer", "default_op")
        self.assertEqual(node_cls.__name__, "PyStudioTCodeViewerNode")


if __name__ == "__main__":
    unittest.main()
