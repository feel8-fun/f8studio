from __future__ import annotations

from typing import Any, Callable

from qtpy import QtCore, QtWidgets

from ..constants import STUDIO_SERVICE_ID

import qtawesome as qta

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
        get_bridge: Callable[[], Any | None],
        get_service_class: Callable[[], str] | None = None,
    ):
        super().__init__(parent)
        self._service_id = str(service_id or "")
        self._get_bridge = get_bridge
        self._get_service_class = get_service_class

        self._btn_toggle = QtWidgets.QToolButton(self)  # start/pause (active)
        self._btn_stop = QtWidgets.QToolButton(self)  # quit process
        self._btn_restart = QtWidgets.QToolButton(self)

        self._play_icon = qta.icon("fa5s.play", color="green")
        self._pause_icon = qta.icon("fa5s.pause", color="yellow")
        self._stop_icon = qta.icon("fa5s.stop", color="red")
        self._restart_icon = qta.icon("fa5s.redo", color="white")


        self._btn_toggle.setAutoRaise(True)
        self._btn_stop.setAutoRaise(True)
        self._btn_restart.setAutoRaise(True)
        self._btn_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self._btn_stop.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self._btn_restart.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)

        self._btn_toggle.clicked.connect(self._on_toggle_clicked)  # type: ignore[attr-defined]
        self._btn_stop.clicked.connect(self._on_stop_clicked)  # type: ignore[attr-defined]
        self._btn_restart.clicked.connect(self._on_restart_clicked)  # type: ignore[attr-defined]

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self._btn_toggle)
        lay.addWidget(self._btn_stop)
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
        self._timer.setInterval(800)
        self._timer.timeout.connect(self.refresh)  # type: ignore[attr-defined]
        self._timer.start()

        self.refresh()

    def _bridge(self) -> Any | None:
        try:
            return self._get_bridge()
        except Exception:
            return None

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

    @QtCore.Slot()
    def refresh(self) -> None:
        sid = str(self._service_id or "").strip()
        bridge = self._bridge()
        enabled = bool(sid) and sid != STUDIO_SERVICE_ID and bridge is not None
        try:
            self.setEnabled(enabled)
        except Exception:
            pass
        if not enabled:
            return

        style = self.style()
        running = self._is_running()
        active = None
        if running:
            try:
                bridge.request_service_status(sid)
            except Exception:
                pass
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

        self._btn_restart.setIcon(self._restart_icon)
        self._btn_restart.setToolTip("Restart service (terminate + deploy + activate)")

        # Button availability.
        self._btn_stop.setEnabled(bool(running))
        self._btn_restart.setEnabled(bool(running))

    def _on_toggle_clicked(self) -> None:
        bridge = self._bridge()
        if bridge is None:
            return
        try:
            sid = str(self._service_id or "").strip()
            if not sid or sid == STUDIO_SERVICE_ID:
                return
            if not self._is_running():
                bridge.start_service_and_deploy(sid, service_class=self._service_class())
                return

            active = None
            try:
                active = bridge.get_cached_service_active(sid)
            except Exception:
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
            bridge.restart_service_and_deploy(sid, service_class=self._service_class())
        finally:
            self.refresh()
