from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Protocol

from qtpy import QtCore, QtWidgets
from NodeGraphQt.nodes.base_node import NodeBaseWidget

from ..nodegraph.operator_basenode import F8StudioOperatorBaseNode
from ..nodegraph.viz_operator_nodeitem import F8StudioVizOperatorNodeItem
from ..ui_bus import UiCommand

logger = logging.getLogger(__name__)
_WEBENGINE_PROFILE_CONFIGURED = False


def _configure_default_webengine_profile() -> None:
    global _WEBENGINE_PROFILE_CONFIGURED
    if _WEBENGINE_PROFILE_CONFIGURED:
        return
    try:
        from PySide6 import QtWebEngineCore  # type: ignore[import-not-found]
    except ImportError:
        logger.exception("failed to import QtWebEngineCore for cache configuration")
        return

    app_data_dir = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.AppDataLocation)
    if not app_data_dir:
        logger.error("Qt AppDataLocation is unavailable; web cache path is not configured")
        return

    cache_dir = Path(app_data_dir) / "webengine_cache"
    storage_dir = Path(app_data_dir) / "webengine_storage"
    cache_dir.mkdir(parents=True, exist_ok=True)
    storage_dir.mkdir(parents=True, exist_ok=True)

    profile = QtWebEngineCore.QWebEngineProfile.defaultProfile()
    profile.setHttpCacheType(QtWebEngineCore.QWebEngineProfile.DiskHttpCache)
    profile.setCachePath(str(cache_dir))
    profile.setPersistentStoragePath(str(storage_dir))
    profile.setPersistentCookiesPolicy(QtWebEngineCore.QWebEngineProfile.ForcePersistentCookies)
    _WEBENGINE_PROFILE_CONFIGURED = True


class _ViewerHandle(Protocol):
    def apply_scene(self, payload: dict[str, Any]) -> None: ...

    def detach_scene(self) -> None: ...


class _Skeleton3DPresenter:
    """
    Pure command router for runtime -> render commands.
    """

    def __init__(self) -> None:
        self.latest_payload: dict[str, Any] | None = None
        self.viewer_open: bool = False
        self._viewer: _ViewerHandle | None = None

    @staticmethod
    def people_count(payload: dict[str, Any] | None) -> int:
        if not isinstance(payload, dict):
            return 0
        people_any = payload.get("people")
        if not isinstance(people_any, list):
            return 0
        return len(people_any)

    def attach_viewer(self, viewer: _ViewerHandle) -> None:
        self._viewer = viewer

    def on_viewer_opened(self) -> None:
        self.viewer_open = True
        payload = self.latest_payload
        if payload is None:
            return
        viewer = self._viewer
        if viewer is None:
            return
        viewer.apply_scene(payload)

    def on_viewer_closed(self) -> None:
        self.viewer_open = False

    def on_set_payload(self, payload: dict[str, Any]) -> None:
        self.latest_payload = payload
        if not self.viewer_open:
            return
        viewer = self._viewer
        if viewer is None:
            return
        viewer.apply_scene(payload)

    def on_detach(self) -> None:
        self.latest_payload = None
        viewer = self._viewer
        if viewer is None:
            return
        viewer.detach_scene()


class _Skeleton3DViewerWindow(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        on_open_state_changed: Callable[[bool], None],
        on_viewer_status_changed: Callable[[str], None],
    ) -> None:
        super().__init__(parent=None)
        self.setWindowTitle("Skeleton3D Viewer")
        self.resize(1200, 760)

        self._on_open_state_changed = on_open_state_changed
        self._on_viewer_status_changed = on_viewer_status_changed
        self._view = None
        self._page_ready = False
        self._pending_payload: dict[str, Any] | None = None
        self._is_open = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        try:
            from PySide6 import QtWebEngineWidgets  # type: ignore[import-not-found]
        except Exception:
            QtWebEngineWidgets = None

        if QtWebEngineWidgets is None:
            fallback = QtWidgets.QLabel("QtWebEngine is not available")
            fallback.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(fallback, 1)
            self._on_viewer_status_changed("QtWebEngine unavailable")
            return

        self._view = QtWebEngineWidgets.QWebEngineView(self)
        _configure_default_webengine_profile()
        self._view.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self._enable_remote_asset_access()
        self._view.loadFinished.connect(self._on_page_loaded)  # type: ignore[attr-defined]
        layout.addWidget(self._view, 1)
        self._load_index_html()
        self._on_viewer_status_changed("loading")

    def _enable_remote_asset_access(self) -> None:
        if self._view is None:
            return
        try:
            from PySide6 import QtWebEngineCore  # type: ignore[import-not-found]

            settings = self._view.settings()
            settings.setAttribute(
                QtWebEngineCore.QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls,
                True,
            )
        except (AttributeError, RuntimeError, TypeError, ImportError):
            logger.exception("failed to enable remote URL access for skeleton3d viewer")

    @staticmethod
    def _asset_dir() -> Path:
        return Path(__file__).resolve().parent / "web_assets" / "viz_three_d"

    def _load_index_html(self) -> None:
        if self._view is None:
            return
        index_path = self._asset_dir() / "index.html"
        if not index_path.exists():
            self._on_viewer_status_changed(f"missing asset: {index_path}")
            return
        self._view.setUrl(QtCore.QUrl.fromLocalFile(str(index_path)))

    def _on_page_loaded(self, ok: bool) -> None:
        self._page_ready = bool(ok)
        if not bool(ok):
            self._on_viewer_status_changed("page load failed")
            return
        self._on_viewer_status_changed("ready")
        pending = self._pending_payload
        self._pending_payload = None
        if pending is not None and self._is_open:
            self._run_set_data(pending)

    def open_viewer(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()
        if self._is_open:
            return
        self._is_open = True
        self._on_open_state_changed(True)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._is_open = False
        self._on_open_state_changed(False)
        super().closeEvent(event)

    def force_shutdown(self) -> None:
        """
        Best-effort shutdown used when the host app is exiting.
        """
        try:
            self.detach_scene()
        except (AttributeError, RuntimeError, TypeError):
            pass
        try:
            self.close()
        except (AttributeError, RuntimeError, TypeError):
            pass
        try:
            self.deleteLater()
        except (AttributeError, RuntimeError, TypeError):
            pass

    def bind_host_parent(self, parent: QtWidgets.QWidget | None) -> None:
        if parent is None:
            return
        if self.parentWidget() is parent:
            return
        try:
            self.setParent(parent, self.windowFlags())
        except Exception:
            logger.exception("failed to bind Skeleton3D viewer parent")

    def apply_scene(self, payload: dict[str, Any]) -> None:
        if not self._is_open:
            return
        if not self._page_ready:
            self._pending_payload = dict(payload)
            return
        self._run_set_data(payload)

    def _run_set_data(self, payload: dict[str, Any]) -> None:
        if self._view is None:
            return
        payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        script = (
            "if (window.Skeleton3DViewer && window.Skeleton3DViewer.setData) {"
            f"window.Skeleton3DViewer.setData({payload_json});"
            "}"
        )
        self._view.page().runJavaScript(script)

    def detach_scene(self) -> None:
        self._pending_payload = None
        if self._view is None:
            return
        if not self._page_ready:
            return
        script = (
            "if (window.Skeleton3DViewer && window.Skeleton3DViewer.detach) {"
            "window.Skeleton3DViewer.detach();"
            "}"
        )
        self._view.page().runJavaScript(script)


class _Skeleton3DControlPane(QtWidgets.QWidget):
    def __init__(self, *, on_open_clicked: Callable[[], None]) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._open_button = QtWidgets.QPushButton("Open Viewer")
        self._open_button.clicked.connect(on_open_clicked)  # type: ignore[arg-type]
        layout.addWidget(self._open_button)

        self.setMinimumWidth(100)
        self.setMinimumHeight(20)
        self.setMaximumWidth(100)
        self.setMaximumHeight(20)

    def set_open_handler(self, on_open_clicked: Callable[[], None]) -> None:
        try:
            self._open_button.clicked.disconnect()
        except (AttributeError, RuntimeError, TypeError):
            pass
        self._open_button.clicked.connect(on_open_clicked)  # type: ignore[arg-type]

    def set_window_open(self, is_open: bool) -> None:
        ...

    def set_viewer_status(self, text: str) -> None:
        ...

    def set_people_count(self, count: int) -> None:
        ...


class _Skeleton3DWidget(NodeBaseWidget):
    def __init__(self, parent=None, name: str = "__skeleton3d", label: str = "") -> None:
        super().__init__(parent=parent, name=name, label=label)
        self._pane = _Skeleton3DControlPane(on_open_clicked=lambda: None)
        self.set_custom_widget(self._pane)

    def get_value(self) -> object:
        return {}

    def set_value(self, value: object) -> None:
        _ = value
        return

    def set_open_handler(self, on_open_clicked: Callable[[], None]) -> None:
        self._pane.set_open_handler(on_open_clicked)

    def set_window_open(self, is_open: bool) -> None:
        self._pane.set_window_open(is_open)

    def set_viewer_status(self, text: str) -> None:
        self._pane.set_viewer_status(text)

    def set_people_count(self, count: int) -> None:
        self._pane.set_people_count(count)


class VizThreeDRenderNode(F8StudioOperatorBaseNode):
    """
    Render node for `f8.viz.three_d`.

    Node body is a compact control panel. 3D rendering lives in a detached viewer
    window, so closing the window only pauses rendering while runtime continues to
    stream/cumulate payload updates.
    """

    def __init__(self):
        super().__init__(qgraphics_item=F8StudioVizOperatorNodeItem)
        self._presenter = _Skeleton3DPresenter()
        self._viewer_window: _Skeleton3DViewerWindow | None = None
        self._app_quit_hook_bound = False
        try:
            widget = _Skeleton3DWidget(self.view, name="__skeleton3d", label="")
            self.add_ephemeral_widget(widget)
            widget.set_open_handler(self._open_viewer)
        except Exception:
            logger.exception("failed to init skeleton3d widget")
        self._bind_app_quit_hook()

    def _bind_app_quit_hook(self) -> None:
        if self._app_quit_hook_bound:
            return
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        try:
            app.aboutToQuit.connect(self._on_app_about_to_quit)  # type: ignore[attr-defined]
            self._app_quit_hook_bound = True
        except Exception:
            logger.exception("failed to bind Skeleton3D app quit hook")

    def _on_app_about_to_quit(self) -> None:
        window = self._viewer_window
        if window is None:
            return
        try:
            window.force_shutdown()
        except Exception:
            logger.exception("failed to shutdown Skeleton3D viewer during app quit")

    def _get_widget(self) -> _Skeleton3DWidget | None:
        try:
            widget = self.get_widget("__skeleton3d")
        except Exception:
            return None
        if not isinstance(widget, _Skeleton3DWidget):
            return None
        return widget

    def _ensure_window(self) -> _Skeleton3DViewerWindow:
        window = self._viewer_window
        if window is not None:
            return window
        window = _Skeleton3DViewerWindow(
            on_open_state_changed=self._on_window_open_state_changed,
            on_viewer_status_changed=self._on_viewer_status_changed,
        )
        self._viewer_window = window
        self._presenter.attach_viewer(window)
        return window

    def _open_viewer(self) -> None:
        window = self._ensure_window()
        app = QtWidgets.QApplication.instance()
        host_parent = app.activeWindow() if app is not None else None
        if isinstance(host_parent, QtWidgets.QWidget):
            window.bind_host_parent(host_parent)
        window.open_viewer()

    def _on_window_open_state_changed(self, is_open: bool) -> None:
        if is_open:
            self._presenter.on_viewer_opened()
        else:
            self._presenter.on_viewer_closed()
        widget = self._get_widget()
        if widget is not None:
            widget.set_window_open(is_open)

    def _on_viewer_status_changed(self, status: str) -> None:
        widget = self._get_widget()
        if widget is not None:
            widget.set_viewer_status(status)

    def apply_ui_command(self, cmd: UiCommand) -> None:
        command = str(cmd.command or "").strip()
        if command not in ("viz.three_d.set", "viz.three_d.detach"):
            return

        if command == "viz.three_d.detach":
            self._presenter.on_detach()
            widget = self._get_widget()
            if widget is not None:
                widget.set_people_count(0)
            return

        try:
            payload = dict(cmd.payload or {})
        except (AttributeError, TypeError, ValueError):
            return

        self._presenter.on_set_payload(payload)
        widget = self._get_widget()
        if widget is not None:
            widget.set_people_count(_Skeleton3DPresenter.people_count(payload))
