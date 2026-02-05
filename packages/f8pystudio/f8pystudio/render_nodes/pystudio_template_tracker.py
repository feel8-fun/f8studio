from __future__ import annotations

import base64
from typing import Any

from qtpy import QtWidgets

from ..command_ui_protocol import CommandUiHandler, CommandUiSource
from ..nodegraph.service_basenode import F8StudioServiceBaseNode
from ..widgets.template_tracker_template import CaptureFrame, TemplateCaptureDialog


class PyStudioTemplateTrackerNode(F8StudioServiceBaseNode, CommandUiHandler):
    def handle_command_ui(
        self,
        cmd: Any,
        *,
        parent: QtWidgets.QWidget | None,
        source: CommandUiSource,
    ) -> bool:
        del source
        call = str(getattr(cmd, "name", "") or "").strip()
        if call != "captureFrame":
            return False
        self._open_template_capture_dialog(parent=parent)
        return True

    def _open_template_capture_dialog(self, *, parent: QtWidgets.QWidget | None) -> None:
        g = None
        try:
            g = self.graph  # NodeGraphQt node graph reference.
        except Exception:
            g = None
        bridge = None
        try:
            bridge = g.service_bridge if g is not None else None
        except Exception:
            bridge = None

        try:
            sid = str(self.id or "").strip()
        except Exception:
            sid = ""

        if bridge is None or not sid:
            return

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
                # Use defaults from the service implementation (params exist in spec, but the
                # custom UI doesn't currently expose them).
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
        dlg.exec()
