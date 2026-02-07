from __future__ import annotations

import json
import logging
from typing import Any, Iterable

from qtpy import QtCore, QtGui, QtWidgets

from f8pysdk import F8OperatorSpec, F8ServiceSpec

from ..nodegraph import F8StudioGraph
from ..nodegraph.session import last_session_path
from ..nodegraph.runtime_compiler import compile_runtime_graphs_from_studio
from ..pystudio_service_bridge import PyStudioServiceBridge, PyStudioServiceBridgeConfig
from ..pystudio_node_registry import SERVICE_CLASS as STUDIO_SERVICE_CLASS
from ..ui_bus import UiCommand, UiCommandApplier
from .node_property_widgets import F8StudioPropertiesBinWidget
from .palette_widget import F8StudioNodesPaletteWidget
from .service_log_widget import ServiceLogDock

logger = logging.getLogger(__name__)


class F8StudioMainWin(QtWidgets.QMainWindow):
    studio_graph: F8StudioGraph

    def __init__(self, node_classes: Iterable[type], parent=None):
        super().__init__(parent)
        self.setWindowTitle("F8PyStudio")
        self.resize(1920, 980)

        self._session_file = last_session_path()

        self.studio_graph = F8StudioGraph()
        self.studio_graph.node_factory.clear_registered_nodes()
        for cls in node_classes:
            self.studio_graph.node_factory.register_node(cls)

        self.setCentralWidget(self.studio_graph.widget)

        self._setup_docks()
        self._setup_menu()
        self._setup_toolbar()
        self._applying_runtime_state = False

        self._bridge = PyStudioServiceBridge(PyStudioServiceBridgeConfig(), parent=self)
        self._bridge.ui_command.connect(self._on_ui_command)  # type: ignore[attr-defined]
        self._bridge.service_output.connect(self._on_service_output)  # type: ignore[attr-defined]
        self._bridge.log.connect(lambda s: self._log_dock.append("studio", str(s) + "\n"))  # type: ignore[attr-defined]
        self._bridge.start()
        try:
            self.studio_graph.set_service_bridge(self._bridge)
        except Exception:
            pass
        self.studio_graph.property_changed.connect(self._on_ui_property_changed)  # type: ignore[attr-defined]

        QtCore.QTimer.singleShot(0, self._auto_load_session)
        QtWidgets.QApplication.instance().aboutToQuit.connect(self._auto_save_session)  # type: ignore[attr-defined]

    @QtCore.Slot(str, str)
    def _on_service_output(self, service_id: str, line: str) -> None:
        try:
            svc_name = str(self._bridge.get_service_class(service_id) or "").strip()
        except Exception:
            svc_name = ""
        if svc_name:
            try:
                self._log_dock.set_service_name(service_id, svc_name)
            except Exception:
                pass
        self._log_dock.append(service_id, line)

    def _setup_docks(self) -> None:
        prop_editor = F8StudioPropertiesBinWidget(node_graph=self.studio_graph)
        self._prop_editor = prop_editor
        prop_dock = QtWidgets.QDockWidget("Properties", self)
        prop_dock.setWidget(prop_editor)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, prop_dock)

        self._log_dock = ServiceLogDock(self)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self._log_dock)

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

        self._deploy_action = QtGui.QAction("Deploy + Run + Monitor", self)
        self._deploy_action.setShortcut("F5")
        self._deploy_action.triggered.connect(self._deploy_run_monitor_action)  # type: ignore[attr-defined]
        menu.addAction(self._deploy_action)

    def _setup_toolbar(self) -> None:
        tb = self.addToolBar("Run")
        tb.setMovable(False)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)

        # Icon button for Deploy + Run + Monitor (F5).
        deploy_icon = self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        self._deploy_action.setIcon(deploy_icon)
        self._deploy_action.setToolTip("Deploy + Run + Monitor (F5)")
        tb.addAction(self._deploy_action)

        tb.addSeparator()

        # Global active/deactive toggle.
        self._active_toggle = QtWidgets.QCheckBox("Services Active", self)
        self._active_toggle.setChecked(True)
        self._active_toggle.setToolTip("Activate/deactivate all managed services (lifecycle control).")
        self._active_toggle.toggled.connect(self._set_global_active)  # type: ignore[attr-defined]

        spacer = QtWidgets.QWidget(self)
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        tb.addWidget(self._active_toggle)

    def _set_global_active(self, active: bool) -> None:
        """
        Global active/deactive for managed services (lifecycle control).
        """
        try:
            active = bool(active)
        except Exception:
            active = True

        try:
            self._bridge.set_managed_active(active)
        except Exception:
            pass
        # Note: studio UI ticking is independent; service lifecycle is remote.

    def closeEvent(self, event):
        self._auto_save_session()
        try:
            self._bridge.stop()
        except Exception:
            pass
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

    def _deploy_run_monitor_action(self) -> None:
        # Deploy implies global active by default.
        try:
            if not self._active_toggle.isChecked():
                self._active_toggle.setChecked(True)
        except Exception:
            pass
        compiled = compile_runtime_graphs_from_studio(self.studio_graph)
        self._bridge.deploy_run_and_monitor(compiled)

    def _on_runtime_state_updated(self, service_id: str, node_id: str, field: str, value: Any, ts_ms: Any) -> None:
        """
        Apply live state updates to the corresponding UI node property (best-effort).
        """
        try:
            node = self.studio_graph.get_node_by_id(str(node_id))
        except Exception:
            node = None
        if node is None:
            return
        try:
            if field in node.model.properties or field in node.model.custom_properties:
                self._applying_runtime_state = True
                try:
                    node.set_property(field, value, push_undo=False)
                    # If this node is currently shown in the Properties dock, also refresh the widget value.
                    try:
                        editor = self._prop_editor.get_property_editor_widget(node)
                        w = editor.get_widget(field) if editor is not None else None
                        if w is not None:
                            try:
                                w.blockSignals(True)
                            except Exception:
                                pass
                            try:
                                w.set_value(value)
                            finally:
                                try:
                                    w.blockSignals(False)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                finally:
                    self._applying_runtime_state = False
        except Exception:
            return

    def _on_ui_command(self, cmd: UiCommand) -> None:
        if str(cmd.command) == "state.update":
            payload = dict(cmd.payload or {})
            field = str(payload.get("field") or "")
            value = payload.get("value")
            service_id = str(payload.get("serviceId") or "")
            node_id = str(cmd.node_id or "")
            if node_id and field:
                self._on_runtime_state_updated(service_id, node_id, field, value, cmd.ts_ms)
            return

        node_id = str(cmd.node_id or "").strip()
        if not node_id:
            return
        try:
            node = self.studio_graph.get_node_by_id(node_id)
        except Exception:
            node = None
        if node is None:
            return
        try:
            if isinstance(node, UiCommandApplier):
                node.apply_ui_command(cmd)
        except Exception:
            return

    def _on_ui_property_changed(self, node: Any, name: str, value: Any) -> None:
        """
        Propagate UI state edits into the corresponding runtime.
        """
        if self._applying_runtime_state:
            return
        try:
            spec = node.spec
        except Exception:
            spec = None
        if not isinstance(spec, (F8OperatorSpec, F8ServiceSpec)):
            return
        service_class = str(spec.serviceClass or "")
        # Only state fields are propagated.
        try:
            state_names = {str(s.name or "") for s in (spec.stateFields or [])}
        except Exception:
            state_names = set()
        if str(name) not in state_names:
            return
        try:
            node_id = str(node.id or "")
        except Exception:
            node_id = ""
        if not node_id:
            return
        if service_class == STUDIO_SERVICE_CLASS:
            self._bridge.set_local_state(node_id, str(name), value)
            return

        # Non-studio services: push state to the runtime via `set_state` endpoint.
        # Service nodes represent service instances themselves: node_id == service_id.
        if isinstance(spec, F8ServiceSpec):
            service_id = node_id
        else:
            try:
                service_id = str(node.svcId or "")
            except Exception:
                service_id = ""
        if not service_id:
            return
        self._bridge.set_remote_state(service_id, node_id, str(name), value)
