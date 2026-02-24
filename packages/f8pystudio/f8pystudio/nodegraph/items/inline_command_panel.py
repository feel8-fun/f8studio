from __future__ import annotations

import json
import logging
from typing import Any

from qtpy import QtCore, QtWidgets

from f8pysdk.schema_helpers import schema_default, schema_type

from ...command_ui_protocol import CommandUiHandler, CommandUiSource
from ...widgets.f8_editor_widgets import F8OptionCombo, F8Switch, F8ValueBar, parse_select_pool
from .service_toolbar_host import F8ForceGlobalToolTipFilter

logger = logging.getLogger(__name__)


def _snapshot_selected_node_ids(node_item: Any) -> list[str]:
    try:
        graph = node_item._graph()
    except Exception:
        return []
    if graph is None:
        return []
    try:
        selected_nodes = list(graph.selected_nodes() or [])
    except Exception:
        return []
    out: list[str] = []
    for node in selected_nodes:
        try:
            node_id = str(node.id or "").strip()
        except Exception:
            node_id = ""
        if node_id:
            out.append(node_id)
    return out


def _restore_selected_node_ids(node_item: Any, ids: list[str]) -> None:
    try:
        graph = node_item._graph()
    except Exception:
        return
    if graph is None:
        return
    target_ids = {str(node_id).strip() for node_id in ids if str(node_id).strip()}
    try:
        nodes = list(graph.all_nodes() or [])
    except Exception:
        nodes = []
    for node in nodes:
        try:
            node_id = str(node.id or "").strip()
        except Exception:
            node_id = ""
        if not node_id:
            continue
        try:
            node.set_property("selected", node_id in target_ids, push_undo=False)
        except Exception:
            continue


def _on_command_pressed(node_item: Any, command: Any) -> None:
    selected_ids = _snapshot_selected_node_ids(node_item)
    node_item._invoke_command(command)
    QtCore.QTimer.singleShot(0, lambda: _restore_selected_node_ids(node_item, selected_ids))


def invoke_command(node_item: Any, cmd: Any) -> None:
    """
    Invoke a command declared on the service spec.

    - no params: fire immediately
    - has params: show dialog to collect args
    """
    if isinstance(cmd, dict):
        call = str(cmd.get("name") or "").strip()
    else:
        try:
            call = str(cmd.name or "").strip()
        except Exception:
            call = ""
    if not call:
        return
    bridge = node_item._bridge()
    if bridge is None:
        return
    sid = node_item._service_id()
    if not sid:
        return
    if not node_item._is_service_running():
        return

    # Allow a node to intercept command invocation with custom UI logic.
    try:
        node = node_item._backend_node()
    except Exception:
        node = None
    if isinstance(node, CommandUiHandler):
        parent = None
        try:
            viewer = node_item.viewer()
            parent = viewer.window() if viewer is not None else None
        except Exception:
            parent = None
        try:
            if bool(node.handle_command_ui(cmd, parent=parent, source=CommandUiSource.NODEGRAPH)):
                return
        except Exception:
            node_id = ""
            try:
                node_id = str(node_item.id or "").strip()
            except Exception:
                node_id = ""
            logger.exception("handle_command_ui failed nodeId=%s", node_id)
    if isinstance(cmd, dict):
        params = list(cmd.get("params") or [])
    else:
        try:
            params = list(cmd.params or [])
        except Exception:
            params = []

    if not params:
        try:
            bridge.invoke_remote_command(sid, call, {})
        except Exception:
            logger.exception("invoke_remote_command failed serviceId=%s call=%s", sid, call)
        return

    args = prompt_command_args(node_item, cmd)
    if args is None:
        return
    try:
        bridge.invoke_remote_command(sid, call, args)
    except Exception:
        logger.exception("invoke_remote_command failed serviceId=%s call=%s", sid, call)


def prompt_command_args(node_item: Any, cmd: Any) -> dict[str, Any] | None:
    if isinstance(cmd, dict):
        call = str(cmd.get("name") or "").strip() or "Command"
        params = list(cmd.get("params") or [])
    else:
        try:
            call = str(cmd.name or "").strip() or "Command"
        except Exception:
            call = "Command"
        try:
            params = list(cmd.params or [])
        except Exception:
            params = []
    if not params:
        return {}

    viewer = node_item.viewer()
    parent = viewer.window() if viewer is not None else None

    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle(call)
    dlg.setModal(True)

    form = QtWidgets.QFormLayout()
    form.setContentsMargins(12, 12, 12, 12)
    form.setSpacing(8)

    editors: dict[str, tuple[QtWidgets.QWidget, Any]] = {}

    for param in params:
        if isinstance(param, dict):
            name = str(param.get("name") or "").strip()
            required = bool(param.get("required") or False)
            ui_raw = str(param.get("uiControl") or "").strip()
            ui = ui_raw.lower()
            schema = param.get("valueSchema")
            desc_raw = param.get("description") or ""
        else:
            try:
                name = str(param.name or "").strip()
            except Exception:
                name = ""
            try:
                required = bool(param.required)
            except Exception:
                required = False
            try:
                ui_raw = str(param.uiControl or "").strip()
                ui = ui_raw.lower()
            except Exception:
                ui_raw = ""
                ui = ""
            try:
                schema = param.valueSchema
            except Exception:
                schema = None
            try:
                desc_raw = param.description or ""
            except Exception:
                desc_raw = ""
        if not name:
            continue

        schema_type_value = schema_type(schema) if schema is not None else ""
        schema_type_value = schema_type_value or ""

        enum_items = node_item._schema_enum_items(schema)
        lo, hi = node_item._schema_numeric_range(schema)
        try:
            default_value = schema_default(schema)
        except Exception:
            default_value = None

        label = f"{name} *" if required else name
        tooltip = str(desc_raw or "").strip()

        def _with_tooltip(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
            if tooltip:
                widget.setToolTip(tooltip)
            return widget

        pool_field = parse_select_pool(ui_raw)
        if enum_items or pool_field or ui in {"select", "dropdown", "dropbox", "combo", "combobox"}:
            combo = F8OptionCombo()
            if pool_field:
                node = node_item._backend_node()
                items = []
                if node is not None:
                    try:
                        value = node.get_property(pool_field)
                        if isinstance(value, (list, tuple)):
                            items = [str(item) for item in value]
                    except Exception:
                        items = []
            else:
                items = list(enum_items)
            combo.set_options(items, labels=items)
            if tooltip:
                combo.set_context_tooltip(tooltip)
            if default_value is not None:
                combo.set_value(str(default_value))

            def _get() -> Any:
                value = combo.value()
                return None if value is None else str(value)

            editors[name] = (_with_tooltip(combo), _get)
            form.addRow(label, combo)
            continue

        if schema_type_value == "boolean" or ui in {"switch", "toggle"}:
            switch = F8Switch()
            switch.set_labels("True", "False")
            if default_value is not None:
                switch.set_value(bool(default_value))

            def _get() -> Any:
                return bool(switch.value())

            editors[name] = (_with_tooltip(switch), _get)
            form.addRow(label, switch)
            continue

        if schema_type_value in {"integer", "number"} and ui == "slider":
            is_int = schema_type_value == "integer"
            bar = F8ValueBar(integer=is_int, minimum=0.0, maximum=1.0)
            bar.set_range(lo, hi)
            if default_value is not None:
                bar.set_value(default_value)

            def _get() -> Any:
                value = bar.value()
                return int(value) if is_int else float(value)

            editors[name] = (_with_tooltip(bar), _get)
            form.addRow(label, bar)
            continue

        if schema_type_value == "integer" or ui in {"spinbox", "int"}:
            spin = QtWidgets.QSpinBox()
            if lo is not None:
                spin.setMinimum(int(lo))
            if hi is not None:
                spin.setMaximum(int(hi))
            if default_value is not None:
                try:
                    spin.setValue(int(default_value))
                except (TypeError, ValueError):
                    pass

            def _get() -> Any:
                return int(spin.value())

            editors[name] = (_with_tooltip(spin), _get)
            form.addRow(label, spin)
            continue

        if schema_type_value == "number" or ui in {"doublespinbox", "float"}:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(6)
            if lo is not None:
                spin.setMinimum(float(lo))
            if hi is not None:
                spin.setMaximum(float(hi))
            if default_value is not None:
                try:
                    spin.setValue(float(default_value))
                except (TypeError, ValueError):
                    pass

            def _get() -> Any:
                return float(spin.value())

            editors[name] = (_with_tooltip(spin), _get)
            form.addRow(label, spin)
            continue

        if schema_type_value in {"object", "array", "any"}:
            text_edit = QtWidgets.QPlainTextEdit()
            text_edit.setMinimumHeight(90)
            if default_value is not None:
                try:
                    text_edit.setPlainText(json.dumps(default_value, ensure_ascii=False, indent=2))
                except Exception:
                    text_edit.setPlainText(str(default_value))

            def _get() -> Any:
                txt = str(text_edit.toPlainText() or "").strip()
                if not txt:
                    return None
                try:
                    return json.loads(txt)
                except Exception:
                    return txt

            editors[name] = (_with_tooltip(text_edit), _get)
            form.addRow(label, text_edit)
            continue

        line_edit = QtWidgets.QLineEdit()
        if default_value is not None:
            line_edit.setText("" if default_value is None else str(default_value))

        def _get() -> Any:
            return str(line_edit.text() or "")

        editors[name] = (_with_tooltip(line_edit), _get)
        form.addRow(label, line_edit)

    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    layout = QtWidgets.QVBoxLayout(dlg)
    layout.addLayout(form)
    layout.addWidget(buttons)

    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)

    while True:
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return None
        args: dict[str, Any] = {}
        missing: list[str] = []
        for param in params:
            if isinstance(param, dict):
                param_name = str(param.get("name") or "").strip()
                required = bool(param.get("required") or False)
            else:
                try:
                    param_name = str(param.name or "").strip()
                except Exception:
                    param_name = ""
                try:
                    required = bool(param.required)
                except Exception:
                    required = False
            if not param_name or param_name not in editors:
                continue
            _widget, getter = editors[param_name]
            try:
                value = getter()
            except Exception:
                value = None
            if isinstance(value, str) and value.strip() == "":
                value = None
            if required and value is None:
                missing.append(param_name)
                continue
            if value is not None:
                args[param_name] = value
        if missing:
            QtWidgets.QMessageBox.warning(dlg, "Missing required fields", "Please fill: " + ", ".join(missing))
            continue
        return args


def ensure_inline_command_widget(node_item: Any) -> None:
    node_item._ensure_bridge_process_hook()
    node = node_item._backend_node()
    if node is None:
        return
    try:
        spec = node.spec
    except Exception:
        spec = None

    try:
        commands = list(node.effective_commands() or [])
    except Exception:
        if spec is None:
            commands = []
        else:
            try:
                commands = list(spec.commands or [])
            except Exception:
                commands = []

    visible_commands: list[Any] = []
    for command in commands:
        if isinstance(command, dict):
            show = bool(command.get("showOnNode") or False)
        else:
            try:
                show = bool(command.showOnNode)
            except Exception:
                show = False
        if show:
            visible_commands.append(command)
    enabled = node_item._is_service_running()

    # Rebuild only when command list / enabled state changes.
    try:

        def _cmd_name_desc(command: Any) -> tuple[str, str]:
            if isinstance(command, dict):
                return str(command.get("name") or ""), str(command.get("description") or "")
            try:
                return str(command.name or ""), str(command.description or "")
            except Exception:
                return "", ""

        serial = json.dumps(
            {
                "cmds": [
                    {
                        "name": _cmd_name_desc(command)[0],
                        "desc": _cmd_name_desc(command)[1],
                    }
                    for command in visible_commands
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    except Exception:
        serial = ""

    # Remove if no commands to show.
    if not visible_commands:
        if node_item._cmd_proxy is not None:
            old = None
            try:
                old = node_item._cmd_proxy.widget()
            except Exception:
                old = None
            try:
                node_item._cmd_proxy.setWidget(None)
            except RuntimeError:
                pass
            if old is not None:
                try:
                    old.setParent(None)
                except (AttributeError, RuntimeError, TypeError):
                    pass
                try:
                    old.deleteLater()
                except (AttributeError, RuntimeError, TypeError):
                    pass
            try:
                node_item._cmd_proxy.setParentItem(None)
                if node_item.scene() is not None:
                    node_item.scene().removeItem(node_item._cmd_proxy)
            except RuntimeError:
                pass
            node_item._cmd_proxy = None
            node_item._cmd_widget = None
            node_item._cmd_buttons = []
        return

    if node_item._cmd_proxy is not None and serial and serial == str(node_item._cmd_serial or ""):
        # Keep enable state in sync (service running can change without spec changes).
        for button in list(node_item._cmd_buttons or []):
            try:
                button.setEnabled(bool(enabled))
            except (AttributeError, RuntimeError, TypeError):
                continue
        return

    node_item._cmd_serial = serial

    # Build widget (only when changed).
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    widget.setAttribute(QtCore.Qt.WA_StyledBackground, True)
    widget.setStyleSheet("background: transparent;")

    node_item._cmd_buttons = []
    for command in visible_commands:
        if isinstance(command, dict):
            btn_label = str(command.get("name") or "")
            desc = str(command.get("description") or "").strip()
        else:
            try:
                btn_label = str(command.name or "")
            except Exception:
                btn_label = ""
            try:
                desc = str(command.description or "").strip()
            except Exception:
                desc = ""
        button = QtWidgets.QPushButton(btn_label)
        tooltip_filter = F8ForceGlobalToolTipFilter(button)
        button.installEventFilter(tooltip_filter)
        node_item._tooltip_filters.append(tooltip_filter)
        button.setMinimumHeight(24)
        button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        button.setEnabled(bool(enabled))
        button.setStyleSheet(
            """
            QPushButton {
                color: rgb(235, 235, 235);
                background: rgba(0, 0, 0, 35);
                border: 1px solid rgba(120, 200, 255, 85);
                border-radius: 6px;
                padding: 6px 10px;
                text-align: center;
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(120, 200, 255, 22);
                border-color: rgba(120, 200, 255, 140);
            }
            QPushButton:pressed {
                background: rgba(120, 200, 255, 35);
                border-color: rgba(120, 200, 255, 160);
            }
            QPushButton:disabled {
                color: rgba(235, 235, 235, 110);
                background: rgba(0, 0, 0, 20);
                border-color: rgba(255, 255, 255, 18);
            }
            """
        )
        if not enabled:
            button.setToolTip((desc + "\n" if desc else "") + "Service not running")
        elif desc:
            button.setToolTip(desc)
        button.pressed.connect(lambda _c=command: _on_command_pressed(node_item, _c))  # type: ignore[attr-defined]
        layout.addWidget(button)
        node_item._cmd_buttons.append(button)

    if node_item._cmd_proxy is None:
        proxy = QtWidgets.QGraphicsProxyWidget(node_item)
        proxy.setWidget(widget)
        proxy.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
        node_item._cmd_proxy = proxy
    else:
        old = None
        try:
            old = node_item._cmd_proxy.widget()
        except Exception:
            old = None
        node_item._cmd_proxy.setWidget(widget)
        if old is not None and old is not widget:
            try:
                old.setParent(None)
            except (AttributeError, RuntimeError, TypeError):
                pass
            try:
                old.deleteLater()
            except (AttributeError, RuntimeError, TypeError):
                pass
    node_item._cmd_widget = widget
