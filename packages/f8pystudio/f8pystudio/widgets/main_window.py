from __future__ import annotations

import json
import logging
from typing import Iterable

from qtpy import QtCore, QtWidgets

from ..nodegraph import F8StudioGraph
from ..nodegraph.session import last_session_path
from ..nodegraph.runtime_compiler import compile_runtime_graphs_from_studio
from .node_property_widgets import F8StudioPropertiesBinWidget
from .palette_widget import F8StudioNodesPaletteWidget

logger = logging.getLogger(__name__)


class F8StudioMainWin(QtWidgets.QMainWindow):
    studio_graph: F8StudioGraph

    def __init__(self, node_classes: Iterable[type], parent=None):
        super().__init__(parent)
        self.setWindowTitle("F8PyStudio")
        self.resize(1200, 800)

        self._session_file = last_session_path()

        self.studio_graph = F8StudioGraph()
        self.studio_graph.node_factory.clear_registered_nodes()
        for cls in node_classes:
            self.studio_graph.node_factory.register_node(cls)

        self.setCentralWidget(self.studio_graph.widget)

        self._setup_docks()
        self._setup_menu()

        QtCore.QTimer.singleShot(0, self._auto_load_session)
        QtWidgets.QApplication.instance().aboutToQuit.connect(self._auto_save_session)  # type: ignore[attr-defined]

    def _setup_docks(self) -> None:
        prop_editor = F8StudioPropertiesBinWidget(node_graph=self.studio_graph)
        prop_dock = QtWidgets.QDockWidget("Properties", self)
        prop_dock.setWidget(prop_editor)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, prop_dock)

        palette = F8StudioNodesPaletteWidget(node_graph=self.studio_graph)
        palette_dock = QtWidgets.QDockWidget("Nodes Palette", self)
        palette_dock.setWidget(palette)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, palette_dock)

    def _setup_menu(self) -> None:
        menu = self.menuBar().addMenu("Graph")

        load_action = QtWidgets.QAction("Load Last Session", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._load_session_action)  # type: ignore[attr-defined]
        menu.addAction(load_action)

        save_action = QtWidgets.QAction("Save Session", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_session_action)  # type: ignore[attr-defined]
        menu.addAction(save_action)

        menu.addSeparator()

        compile_action = QtWidgets.QAction("Compile Runtime Graph (print)", self)
        compile_action.setShortcut("Ctrl+R")
        compile_action.triggered.connect(self._compile_runtime_action)  # type: ignore[attr-defined]
        menu.addAction(compile_action)

    def closeEvent(self, event):
        self._auto_save_session()
        super().closeEvent(event)

    def _auto_load_session(self) -> None:
        loaded = self.studio_graph.load_last_session()
        if loaded:
            logger.info("Loaded session from %s", loaded)

    def _auto_save_session(self) -> None:
        saved = self.studio_graph.save_last_session()
        logger.info("Saved session to %s", saved)

    def _save_session_action(self) -> None:
        path = self.studio_graph.save_last_session()
        QtWidgets.QMessageBox.information(self, "Session saved", f"Saved to:\n{path}")

    def _load_session_action(self) -> None:
        path = self.studio_graph.load_last_session()
        if not path:
            QtWidgets.QMessageBox.information(self, "No session", f"No session file found at:\n{self._session_file}")
            return
        QtWidgets.QMessageBox.information(self, "Session loaded", f"Loaded:\n{path}")

    def _compile_runtime_action(self) -> None:
        compiled = compile_runtime_graphs_from_studio(self.studio_graph)
        payload = compiled.global_graph.model_dump(mode="json", by_alias=True)
        print("\n=== F8Studio RuntimeGraph (global) ===")
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))

        print("\n=== F8Studio RuntimeGraph (per-service) ===")
        for sid, g in compiled.per_service.items():
            p = g.model_dump(mode="json", by_alias=True)
            print(f"\n--- serviceId={sid} ---")
            print(json.dumps(p, ensure_ascii=False, indent=2, default=str))
