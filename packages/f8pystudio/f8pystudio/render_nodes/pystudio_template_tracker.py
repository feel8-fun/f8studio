from __future__ import annotations

import base64
from typing import Any

from qtpy import QtWidgets

from ..nodegraph.service_basenode import F8StudioServiceBaseNode, F8StudioServiceNodeItem
from ..widgets.template_tracker_template import CaptureFrame, TemplateCaptureDialog


class _TemplateTrackerNodeItem(F8StudioServiceNodeItem):
    def _invoke_command(self, cmd: Any) -> None:
        call = str(getattr(cmd, "name", "") or "").strip()
        if call == "captureFrame":
            self._open_template_capture_dialog()
            return
        super()._invoke_command(cmd)

    def _open_template_capture_dialog(self) -> None:
        bridge = self._bridge()
        sid = self._service_id()
        if bridge is None or not sid:
            return

        parent = None
        try:
            v = self.viewer()
            parent = v.window() if v is not None else None
        except Exception:
            parent = None

        def _request_capture(done) -> None:
            def _cb(result: dict[str, Any] | None, err: str | None) -> None:
                if err:
                    done(None, err)
                    return
                if not isinstance(result, dict):
                    done(None, "invalid response")
                    return
                try:
                    img = result.get("image") if isinstance(result.get("image"), dict) else {}
                    b64 = str(img.get("b64") or "")
                    raw = base64.b64decode(b64.encode("ascii"), validate=False) if b64 else b""
                    cap = CaptureFrame(
                        frame_id=int(result.get("frameId") or 0),
                        ts_ms=int(result.get("tsMs") or 0),
                        image_bytes=raw,
                        image_format=str(img.get("format") or ""),
                        width=int(img.get("width") or 0),
                        height=int(img.get("height") or 0),
                    )
                except Exception as exc:
                    done(None, str(exc))
                    return
                done(cap, None)

            try:
                bridge.request_remote_command(sid, "captureFrame", {}, _cb)
            except Exception as exc:
                done(None, str(exc))

        def _set_template_b64(b64: str) -> None:
            bridge.set_remote_state(sid, sid, "templatePngB64", str(b64 or ""))

        dlg = TemplateCaptureDialog(
            parent=parent,
            bridge=bridge,
            service_id=sid,
            request_capture=_request_capture,
            set_template_b64=_set_template_b64,
        )
        try:
            dlg.setAttribute(QtWidgets.QWidget.WA_DeleteOnClose, True)
        except Exception:
            pass
        try:
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
        except Exception:
            dlg.exec()


class PyStudioTemplateTrackerNode(F8StudioServiceBaseNode):
    def __init__(self) -> None:
        super().__init__(qgraphics_item=_TemplateTrackerNodeItem)

