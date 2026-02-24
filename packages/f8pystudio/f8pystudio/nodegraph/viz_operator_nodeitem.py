from __future__ import annotations

from typing import Any

from NodeGraphQt.constants import NodeEnum, PortEnum

from .operator_basenode import F8StudioOperatorNodeItem


class F8StudioVizOperatorNodeItem(F8StudioOperatorNodeItem):
    """
    Viz-style operator node layout.

    Intended for render/viz nodes where:
    - state controls are placed first (top-to-bottom)
    - embedded widgets come after state
    - data/exec/other ports are aligned alongside the widget region (no port rows)
    """

    @staticmethod
    def _port_name(port) -> str:
        try:
            return str(port.name() or "")
        except (AttributeError, TypeError):
            try:
                return str(port.name or "")
            except (AttributeError, TypeError):
                return ""

    @staticmethod
    def _state_field_name_if_visible(state_field) -> str | None:
        """
        Best-effort, explicit access for both dict-style and pydantic/dataclass specs.
        """
        if isinstance(state_field, dict):
            if not bool(state_field.get("showOnNode") or False):
                return None
            name = str(state_field.get("name") or "").strip()
            return name or None
        try:
            if not bool(state_field.showOnNode):
                return None
        except (AttributeError, TypeError):
            return None
        try:
            name = str(state_field.name or "").strip()
        except (AttributeError, TypeError):
            return None
        return name or None

    def _calc_size_horizontal(self):  # type: ignore[override]
        # width, height from node name text.
        text_w = self._text_item.boundingRect().width()
        text_h = self._text_item.boundingRect().height()

        # Determine base port geometry.
        port_width = 0.0
        port_height = 0.0
        for p in list(self.inputs) + list(self.outputs):
            try:
                if not p.isVisible():
                    continue
                port_width = float(p.boundingRect().width())
                port_height = float(p.boundingRect().height())
                break
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue

        # State inline panel heights (span node width).
        state_h = 0.0
        spacing = 1.0
        group_gap = 6.0

        try:
            self._ensure_inline_state_widgets()
        except (AttributeError, RuntimeError, TypeError):
            pass

        state_names: list[str] = []
        try:
            node = self._backend_node()
            eff_states = list(node.effective_state_fields() or []) if node is not None else []
        except Exception:
            eff_states = []
        for s in eff_states:
            nm = self._state_field_name_if_visible(s)
            if nm:
                state_names.append(nm)

        if state_names:
            for sname in state_names:
                header_h = port_height or float(PortEnum.SIZE.value)
                try:
                    header = self._state_inline_headers.get(sname)
                    if header is not None:
                        header_h = float(max(header_h, header.sizeHint().height()))
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass

                body_h = 0.0
                try:
                    body = self._state_inline_bodies.get(sname)
                    if body is not None and body.isVisible():
                        body_h = float(max(0.0, body.sizeHint().height()))
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    body_h = 0.0

                panel_h = header_h + (body_h + spacing if body_h > 0.0 else 0.0)
                state_h += panel_h + spacing
            state_h = max(0.0, state_h - spacing) + group_gap

        # Embedded widgets (eg. plot canvas).
        widget_width = 0.0
        widget_height = 0.0
        for widget in self._widgets.values():
            if not widget.isVisible():
                continue
            try:
                w_rect = widget.boundingRect()
                widget_width = max(widget_width, float(w_rect.width()))
                widget_height += float(w_rect.height())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue

        side_padding = 10.0 if widget_width else 0.0
        width = max(float(NodeEnum.WIDTH.value), float(text_w + 18.0), float(widget_width + side_padding))

        port_region_h = state_h + widget_height
        height = max(float(NodeEnum.HEIGHT.value), float(text_h), float(port_region_h))
        if widget_height:
            height += 10.0

        return width, height

    def _make_state_inline_control(self, state_field: Any):  # type: ignore[override]
        """
        Override a couple of fields for viz nodes:
        - minVal/maxVal: allow blank (auto) via QLineEdit (stores None/float)
        """
        nm = self._state_field_name_if_visible(state_field)
        name = nm or ""
        if name not in {"minVal", "maxVal"}:
            return super()._make_state_inline_control(state_field)

        from qtpy import QtCore, QtWidgets

        def _common_style(w: QtWidgets.QWidget) -> None:
            w.setStyleSheet(
                """
                QLineEdit {
                    color: rgb(235, 235, 235);
                    background: rgba(0, 0, 0, 45);
                    border: 1px solid rgba(255, 255, 255, 55);
                    border-radius: 3px;
                    padding: 1px 4px;
                }
                """
            )

        def _set_node_value(value: Any, *, push_undo: bool) -> None:
            node = self._backend_node()
            if node is None or not name:
                return
            try:
                node.set_property(name, value, push_undo=push_undo)
            except TypeError:
                node.set_property(name, value)

        def _get_node_value() -> Any:
            node = self._backend_node()
            if node is None or not name:
                return None
            try:
                return node.get_property(name)
            except KeyError:
                return None

        line = QtWidgets.QLineEdit()
        line.setMinimumWidth(90)
        line.setPlaceholderText("auto")
        _common_style(line)

        def _apply_value(v: Any) -> None:
            s = "" if v is None else str(v)
            with QtCore.QSignalBlocker(line):
                line.setText(s)

        _apply_value(_get_node_value())
        self._state_inline_updaters[name] = _apply_value

        def _commit() -> None:
            raw = str(line.text() or "").strip()
            if not raw:
                _set_node_value(None, push_undo=True)
                return
            try:
                v = float(raw)
            except ValueError:
                # Keep previous value if parsing fails.
                _apply_value(_get_node_value())
                return
            if v != v:  # NaN
                _set_node_value(None, push_undo=True)
                return
            _set_node_value(v, push_undo=True)

        line.editingFinished.connect(_commit)
        return line

    def _align_viz_state(self, v_offset: float) -> float:
        """
        Align state inline panels top-to-bottom and return the y position after the state block.
        """
        width = float(self._width)
        spacing = 1.0
        group_gap = 6.0

        # Ensure inline widgets exist before aligning so sizing + rows match.
        try:
            self._ensure_inline_state_widgets()
        except (AttributeError, RuntimeError, TypeError):
            pass

        node = self._backend_node()
        if node is None:
            spec = None
        else:
            try:
                spec = node.spec
            except Exception:
                spec = None
        try:
            eff_states = list(node.effective_state_fields() or []) if node is not None else []
        except Exception:
            if spec is None:
                eff_states = []
            else:
                try:
                    eff_states = list(spec.stateFields or [])
                except Exception:
                    eff_states = []

        state_names: list[str] = []
        for s in eff_states:
            nm = self._state_field_name_if_visible(s)
            if nm:
                state_names.append(nm)

        inputs_by_name = {self._port_name(p): p for p in self.inputs if p.isVisible()}
        outputs_by_name = {self._port_name(p): p for p in self.outputs if p.isVisible()}

        port_width = 0.0
        port_height = 0.0
        for p in list(inputs_by_name.values()) + list(outputs_by_name.values()):
            try:
                port_width = float(p.boundingRect().width())
                port_height = float(p.boundingRect().height())
                break
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue

        in_x = (port_width / 2.0) * -1.0
        out_x = width - (port_width / 2.0)

        rect = self.boundingRect()
        inner_x = float(rect.left() + 4.0)
        inner_w = float(max(10.0, rect.width() - 8.0))

        def place_row(in_name: str | None, out_name: str | None, *, y: float) -> None:
            if in_name:
                p = inputs_by_name.get(in_name)
                if p is not None:
                    p.setPos(in_x, y)
            if out_name:
                p = outputs_by_name.get(out_name)
                if p is not None:
                    p.setPos(out_x, y)

        y = float(v_offset)
        for sname in state_names:
            in_name = f"[S]{sname}"
            out_name = f"{sname}[S]"

            panel_proxy = self._state_inline_proxies.get(sname)
            header_h = port_height
            body_h = 0.0
            if panel_proxy is not None and panel_proxy.isVisible():
                try:
                    w = panel_proxy.widget()
                    if w is not None:
                        w.setFixedWidth(int(inner_w))
                        w.adjustSize()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
                try:
                    if self._state_inline_headers.get(sname) is not None:
                        header_h = float(max(port_height, self._state_inline_headers[sname].sizeHint().height()))
                except Exception:
                    header_h = port_height
                try:
                    body_w = self._state_inline_bodies.get(sname)
                    if body_w is not None and body_w.isVisible():
                        body_h = float(max(0.0, body_w.sizeHint().height()))
                except Exception:
                    body_h = 0.0
                try:
                    # Center panels using their actual width, then clamp into node bounds.
                    w = panel_proxy.widget()
                    if w is None:
                        panel_proxy.setPos(inner_x, y)
                    else:
                        panel_w = float(w.width() or 0.0)
                        if panel_w <= 0.0:
                            panel_w = float(panel_proxy.boundingRect().width() or 0.0)
                        if panel_w <= 0.0:
                            panel_proxy.setPos(inner_x, y)
                        else:
                            panel_x = rect.left() + (rect.width() - panel_w) / 2.0
                            min_x = float(inner_x)
                            max_x = float(rect.right() - 4.0 - panel_w)
                            if max_x < min_x:
                                panel_x = min_x
                            else:
                                panel_x = max(min_x, min(panel_x, max_x))
                            panel_proxy.setPos(panel_x, y)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass

            port_y = y + (header_h - port_height) / 2.0 if port_height else y
            place_row(in_name, out_name, y=port_y)

            y += header_h + spacing
            if body_h > 0.0:
                y += body_h + spacing

        if state_names:
            y += group_gap

        self._ports_end_y = y
        return y

    def _align_viz_ports_to_widgets(self, v_offset: float) -> None:
        """
        Align non-state ports along the widget region (left/right of embedded widgets).
        """
        rect = self.boundingRect()

        widget_items = [w for w in self._widgets.values() if w.isVisible()]
        if not widget_items:
            return

        # Widget area in local node coords.
        top = float("inf")
        bottom = float("-inf")
        for w in widget_items:
            try:
                pos_y = float(w.pos().y())
                h = float(w.boundingRect().height())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
            top = min(top, pos_y)
            bottom = max(bottom, pos_y + h)
        if not (top < float("inf") and bottom > float("-inf") and bottom > top):
            return

        # Non-state ports.
        in_ports = []
        out_ports = []
        for p in self.inputs:
            try:
                if not p.isVisible():
                    continue
                if self._port_group(self._port_name(p)) == "state":
                    continue
                in_ports.append(p)
            except (AttributeError, RuntimeError, TypeError):
                continue
        for p in self.outputs:
            try:
                if not p.isVisible():
                    continue
                if self._port_group(self._port_name(p)) == "state":
                    continue
                out_ports.append(p)
            except (AttributeError, RuntimeError, TypeError):
                continue

        if not in_ports and not out_ports:
            return

        # Port geometry.
        port_width = 0.0
        port_height = 0.0
        for p in (in_ports + out_ports):
            try:
                port_width = float(p.boundingRect().width())
                port_height = float(p.boundingRect().height())
                break
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue

        width = float(self._width)
        in_x = (port_width / 2.0) * -1.0
        out_x = width - (port_width / 2.0)

        pad = 8.0
        min_cy = top + pad
        max_cy = bottom - pad
        if max_cy <= min_cy:
            min_cy = top
            max_cy = bottom

        def y_for(index: int, count: int) -> float:
            if count <= 0:
                return min_cy
            t = (index + 1) / (count + 1)
            cy = min_cy + (max_cy - min_cy) * t
            return cy - (port_height / 2.0 if port_height else 0.0)

        for i, p in enumerate(in_ports):
            try:
                p.setPos(in_x, y_for(i, len(in_ports)))
            except (AttributeError, RuntimeError, TypeError):
                continue
        for i, p in enumerate(out_ports):
            try:
                p.setPos(out_x, y_for(i, len(out_ports)))
            except (AttributeError, RuntimeError, TypeError):
                continue

        # Ensure any visible text items follow (but we typically keep them hidden for viz nodes).
        txt_offset = PortEnum.CLICK_FALLOFF.value - 2
        try:
            for port, text in self._input_items.items():
                if port.isVisible():
                    txt_x = port.boundingRect().width() / 2 - txt_offset
                    text.setPos(txt_x, port.y() - 1.5)
            for port, text in self._output_items.items():
                if port.isVisible():
                    txt_width = text.boundingRect().width() - txt_offset
                    txt_x = port.x() - txt_width
                    text.setPos(txt_x, port.y() - 1.5)
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _draw_node_horizontal(self):  # type: ignore[override]
        try:
            self._ensure_inline_state_widgets()
        except (AttributeError, RuntimeError, TypeError):
            pass
        try:
            self._ensure_inline_command_widget()
        except (AttributeError, RuntimeError, TypeError):
            pass

        header_h = float(self._text_item.boundingRect().height() + 4.0)

        # Hide all port text items for viz nodes (ports are color-coded).
        for port, text in self._input_items.items():
            try:
                if port.isVisible():
                    text.setVisible(False)
            except (AttributeError, RuntimeError, TypeError):
                pass
        for port, text in self._output_items.items():
            try:
                if port.isVisible():
                    text.setVisible(False)
            except (AttributeError, RuntimeError, TypeError):
                pass

        # setup initial base size.
        self._set_base_size(add_h=header_h)
        self._set_text_color(self.text_color)
        self._tooltip_disable(self.disabled)

        self.align_label()
        self.align_icon(h_offset=2.0, v_offset=1.0)

        # 1) state first
        self._align_viz_state(v_offset=header_h)
        # 2) then widgets (uses _ports_end_y to avoid overlap)
        self.align_widgets(v_offset=header_h)
        # 3) ports aligned to widget region
        self._align_viz_ports_to_widgets(v_offset=header_h)

        self.update()
