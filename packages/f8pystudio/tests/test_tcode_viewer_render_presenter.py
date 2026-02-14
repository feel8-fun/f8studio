import os
import sys
import unittest
from dataclasses import dataclass, field


PKG_STUDIO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
for p in (PKG_STUDIO, PKG_SDK):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from f8pystudio.render_nodes.pystudio_tcode_viewer import _TCodeViewerPresenter

    _HAS_RENDER_DEPS = True
except Exception:
    _HAS_RENDER_DEPS = False
    _TCodeViewerPresenter = None  # type: ignore[assignment]


@dataclass
class _FakeViewer:
    set_models: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    reset_calls: int = 0
    detach_calls: int = 0

    def set_model(self, model: str) -> None:
        self.set_models.append(str(model))

    def write_tcode(self, line: str) -> None:
        self.writes.append(str(line))

    def reset_viewer(self) -> None:
        self.reset_calls += 1

    def detach_viewer(self) -> None:
        self.detach_calls += 1


@unittest.skipUnless(_HAS_RENDER_DEPS, "render deps unavailable")
class TCodeViewerRenderPresenterTests(unittest.TestCase):
    def test_write_is_dropped_when_viewer_closed(self) -> None:
        assert _TCodeViewerPresenter is not None
        presenter = _TCodeViewerPresenter()
        viewer = _FakeViewer()
        presenter.attach_viewer(viewer)

        presenter.on_write("L0000I500\n")
        self.assertEqual(viewer.writes, [])

    def test_open_viewer_sets_model_without_replay(self) -> None:
        assert _TCodeViewerPresenter is not None
        presenter = _TCodeViewerPresenter()
        viewer = _FakeViewer()
        presenter.attach_viewer(viewer)
        presenter.on_write("L0000I500\n")
        presenter.on_set_model(model="SR6")
        presenter.on_viewer_opened()

        self.assertEqual(viewer.set_models[-1], "SR6")
        self.assertEqual(viewer.reset_calls, 0)
        self.assertEqual(viewer.writes, [])

    def test_detach_notifies_viewer(self) -> None:
        assert _TCodeViewerPresenter is not None
        presenter = _TCodeViewerPresenter()
        viewer = _FakeViewer()
        presenter.attach_viewer(viewer)
        presenter.on_write("L0000I500\n")
        presenter.on_detach()

        self.assertEqual(viewer.detach_calls, 1)


if __name__ == "__main__":
    unittest.main()
