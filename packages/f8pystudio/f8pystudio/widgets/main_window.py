from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from qtpy import QtCore, QtGui, QtWidgets
import qtawesome as qta

from f8pysdk import F8OperatorSpec, F8ServiceSpec

from ..nodegraph import F8StudioGraph
from ..nodegraph.edge_rules import EDGE_KIND_DATA, EDGE_KIND_EXEC, EDGE_KIND_STATE
from ..nodegraph.session import last_session_path
from ..nodegraph.runtime_compiler import compile_runtime_graphs_from_studio
from ..pystudio_service_bridge import PyStudioServiceBridge, PyStudioServiceBridgeConfig
from ..pystudio_node_registry import SERVICE_CLASS as STUDIO_SERVICE_CLASS
from ..ui_notifications import show_info, show_warning
from ..ui_bus import UiCommand, UiCommandApplier
from .node_property_widgets import F8StudioSingleNodePropertiesWidget
from .node_library_widget import F8StudioNodeLibraryWidget
from .service_log_widget import ServiceLogDock

logger = logging.getLogger(__name__)


class F8StudioMainWin(QtWidgets.QMainWindow):
    studio_graph: F8StudioGraph
    _exec_lines_action: QtGui.QAction
    _data_lines_action: QtGui.QAction
    _state_lines_action: QtGui.QAction

    def __init__(self, node_classes: Iterable[type], parent=None):
        super().__init__(parent)
        self.setWindowTitle("F8PyStudio")
        self.resize(1920, 980)

        self._session_file = last_session_path()
        self._session_dialog_dir = str(self._session_file.parent)
        self._exit_autosaved: bool = False

        self.studio_graph = F8StudioGraph()
        self.studio_graph.node_factory.clear_registered_nodes()
        for cls in node_classes:
            self.studio_graph.node_factory.register_node(cls)
        self.studio_graph.install_variant_context_menu_for_nodes(list(node_classes))
        self.studio_graph.install_identity_context_menu_for_nodes(list(node_classes))

        self.setCentralWidget(self.studio_graph.widget)

        self._setup_docks()
        self._deploy_action = self._create_deploy_action()
        self._stop_all_services_action = self._create_stop_all_services_action()
        self._setup_menu()
        self._setup_toolbar()
        self._applying_runtime_state = False

        self._bridge = PyStudioServiceBridge(PyStudioServiceBridgeConfig(), parent=self)
        self._bridge.ui_command.connect(self._on_ui_command)  # type: ignore[attr-defined]
        self._bridge.service_output.connect(self._on_service_output)  # type: ignore[attr-defined]
        self._bridge.service_process_state.connect(self._on_service_process_state)  # type: ignore[attr-defined]
        self._bridge.log.connect(lambda s: self._log_dock.append("studio", str(s) + "\n"))  # type: ignore[attr-defined]
        self._bridge.start()
        try:
            self.studio_graph.set_service_bridge(self._bridge)
        except Exception as exc:
            self._log_dock.report_exception("studio", "studio_graph.set_service_bridge failed", exc)
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
            except (AttributeError, RuntimeError, TypeError):
                pass
        self._log_dock.append(service_id, line)

    @QtCore.Slot(str, bool)
    def _on_service_process_state(self, service_id: str, running: bool) -> None:
        if bool(running):
            return
        try:
            self._log_dock.close_service_tab(service_id)
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _setup_docks(self) -> None:
        prop_editor = F8StudioSingleNodePropertiesWidget(node_graph=self.studio_graph)
        self._prop_editor = prop_editor
        prop_dock = QtWidgets.QDockWidget("Properties", self)
        prop_dock.setWidget(prop_editor)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, prop_dock)

        self._log_dock = ServiceLogDock(self)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self._log_dock)

        node_library = F8StudioNodeLibraryWidget(node_graph=self.studio_graph)
        node_library_dock = QtWidgets.QDockWidget("Node Library", self)
        node_library_dock.setWidget(node_library)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, node_library_dock)

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

        load_from_action = QtWidgets.QAction("Load Session…", self)
        load_from_action.setShortcut("Ctrl+Shift+O")
        load_from_action.triggered.connect(self._load_session_from_action)  # type: ignore[attr-defined]
        menu.addAction(load_from_action)

        save_as_action = QtWidgets.QAction("Save Session As…", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self._save_session_as_action)  # type: ignore[attr-defined]
        menu.addAction(save_as_action)

        menu.addSeparator()

        compile_action = QtWidgets.QAction("Compile Runtime Graph (print)", self)
        compile_action.setShortcut("Ctrl+R")
        compile_action.triggered.connect(self._compile_runtime_action)  # type: ignore[attr-defined]
        menu.addAction(compile_action)

        menu.addAction(self._deploy_action)
        menu.addAction(self._stop_all_services_action)

    def _create_deploy_action(self) -> QtGui.QAction:
        deploy_action = QtGui.QAction("Send Graph", self)
        deploy_action.setShortcut("F5")
        deploy_action.triggered.connect(self._on_deploy_action_triggered)  # type: ignore[attr-defined]
        return deploy_action

    def _create_stop_all_services_action(self) -> QtGui.QAction:
        action = QtGui.QAction("Stop All Services", self)
        action.triggered.connect(self._on_stop_all_services_triggered)  # type: ignore[attr-defined]
        return action

    def _setup_toolbar(self) -> None:
        tb = self.addToolBar("Run")
        tb.setMovable(False)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)

        # Graph file management.
        self._open_icon = qta.icon("fa5s.folder-open", color="white")
        self._save_icon = qta.icon("fa5s.save", color="white")

        self._load_from_action = QtGui.QAction("Load Session…", self)
        self._load_from_action.setIcon(self._open_icon)
        self._load_from_action.setToolTip("Load session from file… (Ctrl+Shift+O)")
        self._load_from_action.triggered.connect(self._load_session_from_action)  # type: ignore[attr-defined]
        tb.addAction(self._load_from_action)

        self._save_as_action = QtGui.QAction("Save Session As…", self)
        self._save_as_action.setIcon(self._save_icon)
        self._save_as_action.setToolTip("Save session to file… (Ctrl+Shift+S)")
        self._save_as_action.triggered.connect(self._save_session_as_action)  # type: ignore[attr-defined]
        tb.addAction(self._save_as_action)

        tb.addSeparator()

        # Send Graph(F5).
        self._send_icon = qta.icon("mdi6.send", color="lightgreen")
        self._deploy_action.setIcon(self._send_icon)
        self._deploy_action.setToolTip("Send graph to services (F5)")
        tb.addAction(self._deploy_action)
        self._stop_all_services_action.setIcon(qta.icon("mdi.stop", color="pink"))
        self._stop_all_services_action.setToolTip("Stop all service processes in graph")
        tb.addAction(self._stop_all_services_action)

        tb.addSeparator()

        self._exec_lines_action = QtGui.QAction("EXEC", self)
        self._exec_lines_action.setCheckable(True)
        self._exec_lines_action.setChecked(True)
        self._exec_lines_action.toggled.connect(self._on_exec_lines_toggled)  # type: ignore[attr-defined]
        self._set_edge_visibility_action_icon(self._exec_lines_action, True)
        tb.addAction(self._exec_lines_action)
        self._set_action_text_beside_icon(tb, self._exec_lines_action, italic=True)

        self._data_lines_action = QtGui.QAction("DATA", self)
        self._data_lines_action.setCheckable(True)
        self._data_lines_action.setChecked(True)
        self._data_lines_action.toggled.connect(self._on_data_lines_toggled)  # type: ignore[attr-defined]
        self._set_edge_visibility_action_icon(self._data_lines_action, True)
        tb.addAction(self._data_lines_action)
        self._set_action_text_beside_icon(tb, self._data_lines_action, italic=True)

        self._state_lines_action = QtGui.QAction("STATE", self)
        self._state_lines_action.setCheckable(True)
        self._state_lines_action.setChecked(True)
        self._state_lines_action.toggled.connect(self._on_state_lines_toggled)  # type: ignore[attr-defined]
        self._set_edge_visibility_action_icon(self._state_lines_action, True)
        tb.addAction(self._state_lines_action)
        self._set_action_text_beside_icon(tb, self._state_lines_action, italic=True)

    def _set_action_text_beside_icon(
        self, toolbar: QtWidgets.QToolBar, action: QtGui.QAction, italic: bool = False
    ) -> None:
        action_widget = toolbar.widgetForAction(action)
        if isinstance(action_widget, QtWidgets.QToolButton):
            action_widget.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            if italic:
                font = QtGui.QFont(action_widget.font())
                font.setItalic(True)
                action_widget.setFont(font)

    def _set_edge_visibility_action_icon(self, action: QtGui.QAction, visible: bool) -> None:
        icon_name = "fa5s.eye" if visible else "fa5s.eye-slash"
        action.setIcon(qta.icon(icon_name, color="white"))

    @QtCore.Slot(bool)
    def _on_exec_lines_toggled(self, checked: bool) -> None:
        visible = bool(checked)
        self.studio_graph.set_edge_kind_visible(EDGE_KIND_EXEC, visible)
        self._set_edge_visibility_action_icon(self._exec_lines_action, visible)

    @QtCore.Slot(bool)
    def _on_data_lines_toggled(self, checked: bool) -> None:
        visible = bool(checked)
        self.studio_graph.set_edge_kind_visible(EDGE_KIND_DATA, visible)
        self._set_edge_visibility_action_icon(self._data_lines_action, visible)

    @QtCore.Slot(bool)
    def _on_state_lines_toggled(self, checked: bool) -> None:
        visible = bool(checked)
        self.studio_graph.set_edge_kind_visible(EDGE_KIND_STATE, visible)
        self._set_edge_visibility_action_icon(self._state_lines_action, visible)

    def closeEvent(self, event):
        self._auto_save_session()
        try:
            self._bridge.stop()
        except Exception as exc:
            self._log_dock.report_exception("studio", "bridge.stop failed", exc)
        super().closeEvent(event)

    def _auto_load_session(self) -> None:
        try:
            loaded = self.studio_graph.load_last_session()
            if loaded:
                logger.info("Loaded session from %s", loaded)
        except Exception as exc:
            self._log_dock.report_exception("studio", "session auto-load failed", exc)
            logger.exception("Auto-load session failed")

    def _auto_save_session(self) -> None:
        # Called from both `closeEvent` and `QApplication.aboutToQuit`; guard to avoid double-save on exit.
        if self._exit_autosaved:
            return
        try:
            saved = self.studio_graph.save_last_session()
            self._exit_autosaved = True
            logger.info("Saved session to %s", saved)
        except Exception:
            try:
                self._log_dock.append("studio", "[session] auto-save failed\n")
            except (AttributeError, RuntimeError, TypeError):
                pass
            logger.exception("Auto-save session failed")

    def _save_session_action(self) -> None:
        path = self.studio_graph.save_last_session()
        show_info(self, "Session saved", f"Saved to:\n{path}")

    def _load_session_action(self) -> None:
        path = self.studio_graph.load_last_session()
        if not path:
            show_info(self, "No session", f"No session file found at:\n{self._session_file}")
            return
        show_info(self, "Session loaded", f"Loaded:\n{path}")

    def _load_session_from_action(self) -> None:
        try:
            start_dir = str(self._session_dialog_dir or "")
        except Exception:
            start_dir = ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Session",
            start_dir,
            "F8 Studio Session (*.json);;JSON (*.json);;All Files (*)",
        )
        p = str(path or "").strip()
        if not p:
            return
        try:
            self.studio_graph.load_session(p)
            self._session_dialog_dir = str(Path(p).resolve().parent)
            self._log_dock.append("studio", f"[session] loaded: {p}\n")
        except Exception as exc:
            self._log_dock.append("studio", f"[session] load failed: {exc}\n")
            self._log_dock.report_exception("studio", f"session load failed ({p})", exc)
            show_warning(self, "Load failed", f"Failed to load:\n{p}\n\n{exc}")

    def _save_session_as_action(self) -> None:
        try:
            start_dir = str(self._session_dialog_dir or "")
        except Exception:
            start_dir = ""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Session As",
            start_dir,
            "F8 Studio Session (*.json);;JSON (*.json);;All Files (*)",
        )
        p = str(path or "").strip()
        if not p:
            return
        if not p.lower().endswith(".json"):
            p = p + ".json"
        try:
            self.studio_graph.save_session(p)
            self._session_dialog_dir = str(Path(p).resolve().parent)
            self._log_dock.append("studio", f"[session] saved: {p}\n")
        except Exception as exc:
            self._log_dock.append("studio", f"[session] save failed: {exc}\n")
            self._log_dock.report_exception("studio", f"session save failed ({p})", exc)
            show_warning(self, "Save failed", f"Failed to save:\n{p}\n\n{exc}")

    def _compile_runtime_action(self) -> None:
        try:
            compiled = compile_runtime_graphs_from_studio(self.studio_graph)
        except ValueError as exc:
            msg = str(exc or "").strip() or "compile failed"
            self._log_dock.append("studio", f"[compile][blocked] {msg}\n")
            show_warning(self, "Compile blocked", msg)
            return
        except Exception as exc:
            self._log_dock.append("studio", f"[compile][error] {exc}\n")
            self._log_dock.report_exception("studio", "compile runtime graph failed", exc)
            show_warning(self, "Compile failed", str(exc))
            return

        payload = compiled.global_graph.model_dump(mode="json", by_alias=True)
        print("\n=== F8Studio RuntimeGraph (global) ===")
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))

        print("\n=== F8Studio RuntimeGraph (per-service) ===")
        for sid, g in compiled.per_service.items():
            p = g.model_dump(mode="json", by_alias=True)
            print(f"\n--- serviceId={sid} ---")
            print(json.dumps(p, ensure_ascii=False, indent=2, default=str))

    def _on_deploy_action_triggered(self) -> None:
        try:
            compiled = compile_runtime_graphs_from_studio(self.studio_graph)
        except ValueError as exc:
            msg = str(exc or "").strip() or "deploy blocked by invalid graph"
            self._log_dock.append("studio", f"[deploy][blocked] {msg}\n")
            show_warning(self, "Deploy blocked", msg)
            return
        except Exception as exc:
            self._log_dock.append("studio", f"[deploy][error] {exc}\n")
            self._log_dock.report_exception("studio", "deploy compile failed", exc)
            show_warning(self, "Deploy failed", str(exc))
            return
        for warning in list(compiled.warnings or ()):
            self._log_dock.append("studio", f"[compile][warn] {warning}\n")
        self._bridge.deploy(compiled)

    def _on_stop_all_services_triggered(self) -> None:
        service_ids: set[str] = set()
        for node in list(self.studio_graph.all_nodes() or []):
            try:
                spec = node.spec
            except Exception:
                continue
            if not isinstance(spec, F8ServiceSpec):
                continue
            if str(spec.serviceClass or "") == STUDIO_SERVICE_CLASS:
                continue
            try:
                service_id = str(node.id or "").strip()
            except Exception:
                service_id = ""
            if service_id:
                service_ids.add(service_id)

        if not service_ids:
            self._log_dock.append("studio", "[service] no graph service instances to stop\n")
            return

        for service_id in sorted(service_ids):
            try:
                self._bridge.stop_service(service_id)
                self._log_dock.append("studio", f"[service] stop requested: {service_id}\n")
            except Exception as exc:
                self._log_dock.report_exception("studio", f"stop service failed ({service_id})", exc)

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
                            except (AttributeError, RuntimeError, TypeError):
                                pass
                            try:
                                w.set_value(value)
                            finally:
                                try:
                                    w.blockSignals(False)
                                except (AttributeError, RuntimeError, TypeError):
                                    pass
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pass
                finally:
                    self._applying_runtime_state = False
        except (AttributeError, RuntimeError, TypeError):
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
        except (AttributeError, RuntimeError, TypeError, ValueError):
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
