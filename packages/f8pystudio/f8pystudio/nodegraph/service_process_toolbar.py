from __future__ import annotations

from typing import Any, Callable

from qtpy import QtCore, QtWidgets

from ..constants import STUDIO_SERVICE_ID

import qtawesome as qta

from .service_bridge_protocol import ServiceBridge


class ServiceProcessToolbar(QtWidgets.QWidget):
    """
    Small toolbar widget (Start/Pause + Stop + Restart) for service process control.

    This controls the local `ServiceProcessManager` via `PyStudioServiceBridge`.
    """

    def __init__(
        self,
        parent=None,
        *,
        service_id: str,
        get_bridge: Callable[[], ServiceBridge | None],
        get_node: Callable[[], Any | None] | None = None,
        get_service_class: Callable[[], str] | None = None,
        get_compiled_graphs: Callable[[], Any | None] | None = None,
    ):
        super().__init__(parent)
        self._service_id = str(service_id or "")
        self._get_bridge = get_bridge
        self._get_node = get_node
        self._get_service_class = get_service_class
        self._get_compiled_graphs = get_compiled_graphs

        self._btn_disable = QtWidgets.QToolButton(self)
        self._btn_toggle = QtWidgets.QToolButton(self)  # start/pause (active)
        self._btn_stop = QtWidgets.QToolButton(self)  # quit process
        self._btn_sync = QtWidgets.QToolButton(self)  # deploy
        self._btn_restart = QtWidgets.QToolButton(self)

        self._disable_icon = qta.icon("fa5s.toggle-on", color="white")
        self._enable_icon = qta.icon("fa5s.toggle-off", color="white")
        self._play_icon = qta.icon("fa5s.play", color="green")
        self._pause_icon = qta.icon("fa5s.pause", color="yellow")
        self._stop_icon = qta.icon("fa5s.stop", color="red")
        self._sync_icon = qta.icon("fa5s.exchange-alt", color="white")
        self._restart_icon = qta.icon("fa5s.redo", color="white")

        self._btn_disable.setAutoRaise(True)
        self._btn_toggle.setAutoRaise(True)
        self._btn_stop.setAutoRaise(True)
        self._btn_sync.setAutoRaise(True)
        self._btn_restart.setAutoRaise(True)
        self._btn_disable.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self._btn_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self._btn_stop.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self._btn_sync.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self._btn_restart.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)

        # Disable is a local-only studio feature: do NOT depend on service_bridge.
        # Use a plain click handler (not checkable) to avoid QToolButton check-state
        # weirdness inside QGraphicsProxyWidget.
        self._btn_disable.setCheckable(False)
        self._btn_disable.setIcon(self._disable_icon)
        self._btn_disable.setToolTip("Disable node (skip in rungraph + do not auto-start)")
        self._btn_disable.clicked.connect(self._on_disable_clicked)  # type: ignore[attr-defined]

        # Default icons even before first successful refresh (eg. bridge not ready yet).
        self._btn_toggle.setIcon(self._play_icon)
        self._btn_stop.setIcon(self._stop_icon)
        self._btn_sync.setIcon(self._sync_icon)
        self._btn_restart.setIcon(self._restart_icon)

        self._btn_toggle.setToolTip("Start service (deploy + activate)")
        self._btn_stop.setToolTip("Terminate service process")
        self._btn_sync.setToolTip("Deploy current rungraph to service")
        self._btn_restart.setToolTip("Restart service (terminate + deploy + activate)")

        self._btn_toggle.clicked.connect(self._on_toggle_clicked)  # type: ignore[attr-defined]
        self._btn_stop.clicked.connect(self._on_stop_clicked)  # type: ignore[attr-defined]
        self._btn_sync.clicked.connect(self._on_sync_clicked)  # type: ignore[attr-defined]
        self._btn_restart.clicked.connect(self._on_restart_clicked)  # type: ignore[attr-defined]

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self._btn_disable)
        lay.addWidget(self._btn_toggle)
        lay.addWidget(self._btn_stop)
        lay.addWidget(self._btn_sync)
        lay.addWidget(self._btn_restart)

        # Match NodeGraphQt's dark UI: a subtle "badge" container with hover feedback.
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            """
            ServiceProcessToolbar {
              background: rgba(30, 30, 30, 190);
              border: 1px solid rgba(255, 255, 255, 28);
              border-radius: 6px;
              padding: 1px;
            }
            ServiceProcessToolbar QToolButton {
              background: transparent;
              border: 0px;
              padding: 2px;
            }
            ServiceProcessToolbar QToolButton:hover {
              background: rgba(255, 255, 255, 22);
              border-radius: 4px;
            }
            """
        )

        # Poll state so crashes/external stops are reflected.
        self._timer = QtCore.QTimer(self)
        # Keep UI responsive when services are started/stopped outside this process.
        self._timer.setInterval(400)
        self._timer.timeout.connect(self.refresh)  # type: ignore[attr-defined]
        self._timer.start()

        self.refresh()

    def set_service_id(self, service_id: str) -> None:
        self._service_id = str(service_id or "").strip()
        self.refresh()

    def _bridge(self) -> ServiceBridge | None:
        try:
            b = self._get_bridge()
            return b if b is not None else None
        except Exception:
            return None

    def _node(self) -> Any | None:
        try:
            return self._get_node() if self._get_node is not None else None
        except Exception:
            return None

    def _node_item(self) -> Any | None:
        """
        Best-effort access to the QGraphicsItem node view item that owns this toolbar.

        This allows disabling the node locally even when the backend node/bridge isn't available yet.
        """
        try:
            proxy = self.graphicsProxyWidget()
        except Exception:
            proxy = None
        if proxy is None:
            return None
        try:
            return proxy.parentItem()
        except Exception:
            return None

    def _is_node_disabled(self) -> bool:
        n = self._node()
        if n is not None:
            try:
                return bool(n.disabled())
            except (AttributeError, RuntimeError, TypeError):
                pass
            try:
                return bool(n.view.disabled)
            except (AttributeError, RuntimeError, TypeError):
                pass
        # Fallback: use the view item directly.
        item = self._node_item()
        if item is not None:
            try:
                return bool(item.disabled)
            except (AttributeError, RuntimeError, TypeError):
                pass
        return False

    def _set_node_disabled(self, disabled: bool) -> None:
        n = self._node()
        if n is not None:
            try:
                n.set_disabled(bool(disabled))
                return
            except (AttributeError, RuntimeError, TypeError):
                pass
            # Prefer setting backend node state (persists in session); also try the view.
            try:
                n.view.disabled = bool(disabled)
            except (AttributeError, RuntimeError, TypeError):
                pass
        # Fallback: disable the view item directly (local-only).
        item = self._node_item()
        if item is None:
            return
        try:
            item.disabled = bool(disabled)
        except (AttributeError, RuntimeError, TypeError):
            return

    def _is_running(self) -> bool:
        bridge = self._bridge()
        if bridge is None:
            return False
        try:
            return bool(bridge.is_service_running(self._service_id))
        except Exception:
            return False

    def _service_class(self) -> str:
        try:
            return str(self._get_service_class() or "") if self._get_service_class is not None else ""
        except Exception:
            return ""

    def _compiled_graphs(self) -> Any | None:
        try:
            return self._get_compiled_graphs() if self._get_compiled_graphs is not None else None
        except Exception:
            return None

    @QtCore.Slot()
    def refresh(self) -> None:
        sid = str(self._service_id or "").strip()
        enabled = bool(sid) and sid != STUDIO_SERVICE_ID
        if not enabled:
            try:
                self._btn_disable.setEnabled(False)
                self._btn_toggle.setEnabled(False)
                self._btn_stop.setEnabled(False)
                self._btn_sync.setEnabled(False)
                self._btn_restart.setEnabled(False)
            except (AttributeError, RuntimeError, TypeError):
                pass
            return

        # During node creation / graph reload, the toolbar widget can exist briefly
        # before the proxy is in a scene or the backend node is resolvable. This
        # is a transient state; do not stop polling or permanently disable the UI.
        item = self._node_item()
        if item is not None:
            try:
                if item.scene() is None:
                    # Not in scene yet (or being removed). Keep polling.
                    self._btn_disable.setEnabled(True)
                    self._btn_toggle.setEnabled(False)
                    self._btn_stop.setEnabled(False)
                    self._btn_sync.setEnabled(False)
                    self._btn_restart.setEnabled(False)
                    self._btn_toggle.setToolTip("Start service (initializing)")
                    return
            except (AttributeError, RuntimeError, TypeError):
                pass
        if self._node() is None and item is None:
            # Backend graph/node not ready yet (or node was deleted). Keep polling;
            # if the widget is truly orphaned it will be deleted with its proxy.
            try:
                self._btn_disable.setEnabled(False)
                self._btn_toggle.setEnabled(False)
                self._btn_stop.setEnabled(False)
                self._btn_sync.setEnabled(False)
                self._btn_restart.setEnabled(False)
                self._btn_toggle.setToolTip("Start service (node not ready)")
            except (AttributeError, RuntimeError, TypeError):
                pass
            return

        # Disable button works even without a bridge connection.
        try:
            self._btn_disable.setEnabled(True)
        except (AttributeError, RuntimeError, TypeError):
            pass

        disabled = self._is_node_disabled()
        try:
            # Show current state: when disabled -> show "enable" check icon; else show "ban".
            self._btn_disable.setIcon(self._enable_icon if disabled else self._disable_icon)
            self._btn_disable.setToolTip("Enable node" if disabled else "Disable node (skip in rungraph + do not auto-start)")
        except (AttributeError, RuntimeError, TypeError):
            pass

        # When disabled, lock out process controls regardless of bridge availability.
        if disabled:
            try:
                self._btn_toggle.setEnabled(False)
                self._btn_stop.setEnabled(False)
                self._btn_sync.setEnabled(False)
                self._btn_restart.setEnabled(False)
                self._btn_toggle.setToolTip("Disabled")
                self._btn_stop.setToolTip("Disabled")
                self._btn_sync.setToolTip("Disabled")
                self._btn_restart.setToolTip("Disabled")
            except (AttributeError, RuntimeError, TypeError):
                pass
            return

        bridge = self._bridge()

        # If bridge isn't available yet, keep the process buttons visible but disabled.
        if bridge is None:
            try:
                self._btn_toggle.setEnabled(False)
                self._btn_stop.setEnabled(False)
                self._btn_sync.setEnabled(False)
                self._btn_restart.setEnabled(False)
                self._btn_toggle.setToolTip("Start service (bridge not ready)")
            except (AttributeError, RuntimeError, TypeError):
                pass
            return

        try:
            bridge.request_service_status(sid)
        except (AttributeError, RuntimeError, TypeError):
            pass
        try:
            running = bool(bridge.is_service_running(self._service_id))
        except Exception:
            running = False
        active = None
        if running:
            try:
                active = bridge.get_cached_service_active(sid)
            except Exception:
                active = None

        if not running:
            self._btn_toggle.setIcon(self._play_icon)
            self._btn_toggle.setToolTip("Start service (deploy + activate)")
        else:
            if active is False:
                self._btn_toggle.setIcon(self._play_icon)
                self._btn_toggle.setToolTip("Activate service")
            else:
                self._btn_toggle.setIcon(self._pause_icon)
                self._btn_toggle.setToolTip("Deactivate service")

        self._btn_stop.setIcon(self._stop_icon)
        self._btn_stop.setToolTip("Terminate service process")

        self._btn_sync.setIcon(self._sync_icon)
        self._btn_sync.setToolTip("Deploy current rungraph to service")

        self._btn_restart.setIcon(self._restart_icon)
        self._btn_restart.setToolTip("Restart service (terminate + deploy + activate)")

        # Button availability.
        self._btn_stop.setEnabled(bool(running))
        self._btn_sync.setEnabled(bool(running))
        self._btn_restart.setEnabled(bool(running))
        self._btn_toggle.setEnabled(True)

    @QtCore.Slot()
    def _on_disable_clicked(self) -> None:
        cur = self._is_node_disabled()
        nxt = not bool(cur)
        self._set_node_disabled(bool(nxt))
        try:
            self.refresh()
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _on_toggle_clicked(self) -> None:
        bridge = self._bridge()
        if bridge is None:
            return
        try:
            sid = str(self._service_id or "").strip()
            if not sid or sid == STUDIO_SERVICE_ID:
                return
            if not self._is_running():
                compiled = self._compiled_graphs()
                if compiled is not None:
                    bridge.start_service_and_deploy(sid, service_class=self._service_class(), compiled=compiled)
                else:
                    bridge.start_service_and_deploy(sid, service_class=self._service_class())
                return

            active = None
            try:
                active = bridge.get_cached_service_active(sid)
            except (AttributeError, RuntimeError, TypeError):
                active = None
            if active is False:
                bridge.set_service_active(sid, True)
            else:
                bridge.set_service_active(sid, False)
        finally:
            self.refresh()

    def _on_stop_clicked(self) -> None:
        bridge = self._bridge()
        if bridge is None:
            return
        try:
            sid = str(self._service_id or "").strip()
            if not sid or sid == STUDIO_SERVICE_ID:
                return
            bridge.stop_service(sid)
        finally:
            self.refresh()

    def _on_restart_clicked(self) -> None:
        bridge = self._bridge()
        if bridge is None:
            return
        try:
            sid = str(self._service_id or "").strip()
            if not sid or sid == STUDIO_SERVICE_ID:
                return
            compiled = self._compiled_graphs()
            if compiled is not None:
                bridge.restart_service_and_deploy(sid, service_class=self._service_class(), compiled=compiled)
            else:
                bridge.restart_service_and_deploy(sid, service_class=self._service_class())
        finally:
            self.refresh()

    def _on_sync_clicked(self) -> None:
        bridge = self._bridge()
        if bridge is None:
            return
        try:
            sid = str(self._service_id or "").strip()
            if not sid or sid == STUDIO_SERVICE_ID:
                return
            if not self._is_running():
                return
            compiled = self._compiled_graphs()
            if compiled is None:
                return
            bridge.deploy_service_rungraph(sid, compiled=compiled)
        finally:
            self.refresh()
