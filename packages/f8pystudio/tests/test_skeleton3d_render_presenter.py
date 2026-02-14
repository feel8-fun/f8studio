import os
import sys
import unittest
from dataclasses import dataclass, field
from typing import Any


PKG_STUDIO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
for p in (PKG_STUDIO, PKG_SDK):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from f8pystudio.render_nodes.pystudio_skeleton3d import _Skeleton3DPresenter

    _HAS_RENDER_DEPS = True
except Exception:
    _HAS_RENDER_DEPS = False
    _Skeleton3DPresenter = None  # type: ignore[assignment]


@dataclass
class _FakeViewer:
    scenes: list[dict[str, Any]] = field(default_factory=list)
    detach_calls: int = 0

    def apply_scene(self, payload: dict[str, Any]) -> None:
        self.scenes.append(dict(payload))

    def detach_scene(self) -> None:
        self.detach_calls += 1


@unittest.skipUnless(_HAS_RENDER_DEPS, "render deps unavailable")
class Skeleton3DRenderPresenterTests(unittest.TestCase):
    def test_set_payload_does_not_render_when_viewer_closed(self) -> None:
        assert _Skeleton3DPresenter is not None
        presenter = _Skeleton3DPresenter()
        viewer = _FakeViewer()
        presenter.attach_viewer(viewer)

        presenter.on_set_payload({"people": [{"name": "A"}]})
        self.assertEqual(viewer.scenes, [])
        self.assertIsNotNone(presenter.latest_payload)

    def test_open_viewer_consumes_latest_payload(self) -> None:
        assert _Skeleton3DPresenter is not None
        presenter = _Skeleton3DPresenter()
        viewer = _FakeViewer()
        presenter.attach_viewer(viewer)

        payload = {"people": [{"name": "A"}, {"name": "B"}]}
        presenter.on_set_payload(payload)
        presenter.on_viewer_opened()
        self.assertEqual(len(viewer.scenes), 1)
        self.assertEqual(viewer.scenes[0], payload)

    def test_detach_clears_cache_and_notifies_viewer(self) -> None:
        assert _Skeleton3DPresenter is not None
        presenter = _Skeleton3DPresenter()
        viewer = _FakeViewer()
        presenter.attach_viewer(viewer)

        presenter.on_set_payload({"people": [{"name": "A"}]})
        presenter.on_viewer_opened()
        presenter.on_detach()
        self.assertIsNone(presenter.latest_payload)
        self.assertEqual(viewer.detach_calls, 1)


if __name__ == "__main__":
    unittest.main()
