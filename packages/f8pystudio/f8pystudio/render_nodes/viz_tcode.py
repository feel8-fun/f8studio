from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Protocol

from NodeGraphQt.nodes.base_node import NodeBaseWidget
from qtpy import QtCore, QtWidgets

from ..nodegraph.operator_basenode import F8StudioOperatorBaseNode
from ..nodegraph.viz_operator_nodeitem import F8StudioVizOperatorNodeItem
from ..ui_bus import UiCommand

logger = logging.getLogger(__name__)


class _ViewerHandle(Protocol):
    def set_model(self, model: str) -> None: ...

    def write_tcode(self, line: str) -> None: ...

    def reset_viewer(self) -> None: ...

    def detach_viewer(self) -> None: ...


class _TCodeViewerPresenter:
    """
    Pure command router from runtime UI commands to viewer window.
    """

    def __init__(self) -> None:
        self.viewer_open: bool = False
        self.model: str = "SR6"
        self._viewer: _ViewerHandle | None = None

    def attach_viewer(self, viewer: _ViewerHandle) -> None:
        self._viewer = viewer

    def on_viewer_opened(self) -> None:
        self.viewer_open = True
        viewer = self._viewer
        if viewer is None:
            return
        viewer.set_model(self.model)

    def on_viewer_closed(self) -> None:
        self.viewer_open = False

    def on_set_model(self, *, model: str) -> None:
        self.model = str(model or "SR6")
        viewer = self._viewer
        if not self.viewer_open or viewer is None:
            return
        viewer.set_model(self.model)
        viewer.reset_viewer()

    def on_write(self, line: str) -> None:
        viewer = self._viewer
        if not self.viewer_open or viewer is None:
            return
        viewer.write_tcode(line)

    def on_reset(self) -> None:
        viewer = self._viewer
        if not self.viewer_open or viewer is None:
            return
        viewer.reset_viewer()

    def on_detach(self) -> None:
        viewer = self._viewer
        if viewer is None:
            return
        viewer.detach_viewer()


class _TCodeViewerWindow(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        on_open_state_changed: Callable[[bool], None],
    ) -> None:
        super().__init__(parent=None)
        self.setWindowTitle("TCode Viewer")
        self.resize(1080, 720)
        self._on_open_state_changed = on_open_state_changed
        self._view = None
        self._page_ready = False
        self._is_open = False
        self._pending_scripts: list[str] = []

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
            return

        self._view = QtWebEngineWidgets.QWebEngineView(self)
        self._view.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self._view.loadFinished.connect(self._on_page_loaded)  # type: ignore[attr-defined]
        layout.addWidget(self._view, 1)
        self._load_index_html()

    @staticmethod
    def _asset_dir() -> Path:
        return Path(__file__).resolve().parent / "web_assets" / "tcode_viewer"

    def _load_index_html(self) -> None:
        if self._view is None:
            return
        index_path = self._asset_dir() / "index.html"
        if not index_path.exists():
            logger.error("tcode viewer asset missing: %s", index_path)
            return
        self._view.setUrl(QtCore.QUrl.fromLocalFile(str(index_path)))

    def _on_page_loaded(self, ok: bool) -> None:
        self._page_ready = bool(ok)
        if not bool(ok):
            logger.error("tcode viewer page load failed")
            return
        if not self._is_open:
            return
        scripts = list(self._pending_scripts)
        self._pending_scripts = []
        for script in scripts:
            self._run_script(script)

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
        try:
            self.detach_viewer()
        except Exception:
            logger.exception("failed to detach tcode viewer before shutdown")
        try:
            self.close()
        except Exception:
            logger.exception("failed to close tcode viewer during shutdown")
        try:
            self.deleteLater()
        except Exception:
            logger.exception("failed to schedule deleteLater for tcode viewer")

    def bind_host_parent(self, parent: QtWidgets.QWidget | None) -> None:
        if parent is None:
            return
        if self.parentWidget() is parent:
            return
        try:
            self.setParent(parent, self.windowFlags())
        except Exception:
            logger.exception("failed to bind TCode viewer parent")

    def set_model(self, model: str) -> None:
        script = f"window.TCodeViewer?.setModel({json.dumps(str(model))});"
        self._queue_or_run(script)

    def write_tcode(self, line: str) -> None:
        script = f"window.TCodeViewer?.writeTCode({json.dumps(str(line))});"
        self._queue_or_run(script)

    def reset_viewer(self) -> None:
        self._queue_or_run("window.TCodeViewer?.resetViewer();")

    def detach_viewer(self) -> None:
        self._pending_scripts = []
        self._queue_or_run("window.TCodeViewer?.detachViewer();")

    def _queue_or_run(self, script: str) -> None:
        if not self._is_open:
            return
        if not self._page_ready:
            self._pending_scripts.append(script)
            return
        self._run_script(script)

    def _run_script(self, script: str) -> None:
        if self._view is None:
            return
        self._view.page().runJavaScript(script)


class _TCodeViewerControlPane(QtWidgets.QWidget):
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
        self.setMaximumWidth(150)
        self.setMaximumHeight(40)

    def set_open_handler(self, on_open_clicked: Callable[[], None]) -> None:
        try:
            self._open_button.clicked.disconnect()
        except (TypeError, RuntimeError):
            logger.debug("no previous open handler to disconnect")
        self._open_button.clicked.connect(on_open_clicked)  # type: ignore[arg-type]

    def set_window_open(self, is_open: bool) -> None:
        ...


class _TCodeViewerWidget(NodeBaseWidget):
    def __init__(self, parent=None, name: str = "__tcode_viewer", label: str = "") -> None:
        super().__init__(parent=parent, name=name, label=label)
        self._pane = _TCodeViewerControlPane(on_open_clicked=lambda: None)
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


class VizTCodeRenderNode(F8StudioOperatorBaseNode):
    def __init__(self) -> None:
        super().__init__(qgraphics_item=F8StudioVizOperatorNodeItem)
        self._presenter = _TCodeViewerPresenter()
        self._viewer_window: _TCodeViewerWindow | None = None
        self._app_quit_hook_bound = False
        try:
            widget = _TCodeViewerWidget(self.view, name="__tcode_viewer", label="")
            self.add_ephemeral_widget(widget)
            widget.set_open_handler(self._open_viewer)
        except Exception:
            logger.exception("failed to init tcode viewer widget")
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
            logger.exception("failed to bind TCode viewer app quit hook")

    def _on_app_about_to_quit(self) -> None:
        window = self._viewer_window
        if window is None:
            return
        try:
            window.force_shutdown()
        except Exception:
            logger.exception("failed to shutdown TCode viewer during app quit")

    def _get_widget(self) -> _TCodeViewerWidget | None:
        try:
            widget = self.get_widget("__tcode_viewer")
        except Exception:
            return None
        if not isinstance(widget, _TCodeViewerWidget):
            return None
        return widget

    def _ensure_window(self) -> _TCodeViewerWindow:
        window = self._viewer_window
        if window is not None:
            return window
        window = _TCodeViewerWindow(
            on_open_state_changed=self._on_window_open_state_changed,
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

    def apply_ui_command(self, cmd: UiCommand) -> None:
        command = str(cmd.command or "").strip()
        if command not in (
            "viz.tcode.set_model",
            "viz.tcode.write",
            "viz.tcode.reset",
            "viz.tcode.detach",
        ):
            return

        if command == "viz.tcode.detach":
            self._presenter.on_detach()
            return

        payload = dict(cmd.payload or {})
        if command == "viz.tcode.set_model":
            model = str(payload.get("model") or "SR6")
            self._presenter.on_set_model(model=model)
            return

        if command == "viz.tcode.reset":
            self._presenter.on_reset()
            return

        line_any = payload.get("line")
        if not isinstance(line_any, str):
            return
        self._presenter.on_write(line_any)
