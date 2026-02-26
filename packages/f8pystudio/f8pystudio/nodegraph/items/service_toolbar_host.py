from __future__ import annotations

from qtpy import QtCore, QtGui, QtWidgets


class F8ElideToolButton(QtWidgets.QToolButton):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._full_text = ""

    def set_full_text(self, text: str) -> None:
        self._full_text = str(text or "")
        self._apply_elide()

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_elide()

    def event(self, event):  # type: ignore[override]
        # Tooltips on embedded widgets inside a QGraphicsProxyWidget can pick up
        # an unexpected palette/style (showing as a black box). Force the
        # tooltip to be shown with the global/default styling by passing
        # widget=None.
        try:
            if event.type() == QtCore.QEvent.ToolTip:
                tip = str(self.toolTip() or "").strip()
                if not tip:
                    return True
                pos = None
                try:
                    pos = event.globalPos()
                except AttributeError:
                    try:
                        pos = event.globalPosition().toPoint()
                    except (AttributeError, RuntimeError, TypeError):
                        pos = None
                if pos is not None:
                    QtWidgets.QToolTip.showText(pos, tip, None)
                    return True
        except (AttributeError, RuntimeError, TypeError):
            pass
        return super().event(event)

    def _apply_elide(self) -> None:
        try:
            fm = QtGui.QFontMetrics(self.font())
            # Leave room for the arrow icon.
            width = max(10, int(self.width() - 24))
            self.setText(fm.elidedText(self._full_text, QtCore.Qt.ElideRight, width))
        except RuntimeError:
            self.setText(self._full_text)


class F8ForceGlobalToolTipFilter(QtCore.QObject):
    """
    Force tooltip display via `QToolTip.showText(..., widget=None)` to avoid
    dark/black tooltip palette issues when widgets are embedded in a
    `QGraphicsProxyWidget`.
    """

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # type: ignore[override]
        if event.type() != QtCore.QEvent.ToolTip:
            return super().eventFilter(watched, event)
        if not isinstance(watched, QtWidgets.QWidget):
            return True
        tip = str(watched.toolTip() or "").strip()
        if not tip:
            return True
        try:
            pos = event.globalPos()  # type: ignore[attr-defined]
        except Exception:
            try:
                pos = event.globalPosition().toPoint()  # type: ignore[attr-defined]
            except Exception:
                return True
        QtWidgets.QToolTip.showText(pos, tip, None)
        return True
