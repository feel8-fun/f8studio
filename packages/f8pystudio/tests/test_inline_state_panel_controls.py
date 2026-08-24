from __future__ import annotations

import math
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets
from NodeGraphQt.custom_widgets.properties_bin.node_property_factory import NodePropertyWidgetFactory

from f8pysdk.codec import copy_model
from f8pysdk.specs import F8StateAccess, F8StateSpec, array_schema, integer_schema, number_schema, string_schema
from f8pystudio.nodegraph.items.state_inline_controls import (
    build_state_inline_control,
    ensure_state_inline_controls,
    state_inline_control_serial,
    sync_state_inline_controls_from_graph_property,
)
from f8pystudio.nodegraph.items.node_item_core import StateFieldInfo
from f8pystudio.nodegraph.items.service_toolbar_host import F8ForceGlobalToolTipFilter
from f8pystudio.ui.components.controls import F8Dial, F8OptionCombo, F8RangeBar
from f8pystudio.ui.components.state_editors import (
    F8CodeButtonEditor,
    F8DialEditor,
    F8IncrementButtonEditor,
    F8RangeBarEditor,
    F8WrapLineEditor,
)
from f8pystudio.ui.support.state_panel_controls import build_state_panel_control
from f8pystudio.nodegraph.state_schema import schema_numeric_range
from f8pystudio.ui.components.wave import (
    WaveHeatmapControl,
    WavePatternEditorControl,
    WavePreviewControl,
    graph_draw_rect,
    point_to_widget_pos,
)


class _FakeBackendNode:
    def __init__(self, props: dict[str, Any]) -> None:
        self._props = dict(props)
        self.spec = None
        self.id = "nodeA"

    def get_property(self, name: str) -> Any:
        return self._props.get(str(name), None)

    def set_property(self, name: str, value: Any, *, push_undo: bool = True) -> None:
        del push_undo
        self._props[str(name)] = value


class _FakeNodeItem:
    def __init__(self, *, code_value: str) -> None:
        self.id = "nodeA"
        self.name = "nodeA"
        self._backend = _FakeBackendNode({"code": code_value})
        self._state_inline_updaters: dict[str, Any] = {}
        self._state_inline_option_pools: dict[str, str] = {}
        self._tooltip_filters: list[Any] = []
        self._open_code_editors: list[QtWidgets.QDialog] = []

    def _schema_enum_items(self, schema: Any) -> list[str]:
        del schema
        return []

    def _schema_numeric_range(self, schema: Any) -> tuple[float | None, float | None]:
        del schema
        return None, None

    def _is_state_inline_input_connected(self, field_name: str) -> bool:
        del field_name
        return False

    def _backend_node(self) -> _FakeBackendNode:
        return self._backend


class _SerialNodeItem(_FakeNodeItem):
    def __init__(self) -> None:
        super().__init__(code_value="")

    def _schema_numeric_range(self, schema: Any) -> tuple[float | None, float | None]:
        return schema_numeric_range(schema)


def _ensure_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is not None:
        return app
    return QtWidgets.QApplication([])


def _code_field() -> StateFieldInfo:
    return StateFieldInfo(
        name="code",
        label="Code",
        tooltip="Python source code.",
        show_on_node=True,
        access="rw",
        access_str="rw",
        required=True,
        ui_control="code",
        ui_language="python",
        value_schema=None,
    )


def _wave_preview_field() -> StateFieldInfo:
    return StateFieldInfo(
        name="preview",
        label="Preview",
        tooltip="Preview waveform.",
        show_on_node=True,
        access="ro",
        access_str="ro",
        required=True,
        ui_control="wave_preview",
        ui_language=None,
        value_schema=None,
    )


def _wave_heatmap_field() -> StateFieldInfo:
    return StateFieldInfo(
        name="heatmap",
        label="Heatmap",
        tooltip="Wave heatmap.",
        show_on_node=True,
        access="ro",
        access_str="ro",
        required=True,
        ui_control="wave_heatmap",
        ui_language=None,
        value_schema=None,
    )


def _selected_axis_field() -> StateFieldInfo:
    return StateFieldInfo(
        name="selectedAxis",
        label="Selected Axis",
        tooltip="Axis selector.",
        show_on_node=True,
        access="rw",
        access_str="rw",
        required=True,
        ui_control="select[allAxes]",
        ui_language=None,
        value_schema=None,
    )


def _wave_pattern_field() -> StateFieldInfo:
    return StateFieldInfo(
        name="points",
        label="Points",
        tooltip="Editable control points.",
        show_on_node=True,
        access="rw",
        access_str="rw",
        required=True,
        ui_control="wave_pattern_editor",
        ui_language=None,
        value_schema=None,
    )


def _button_field() -> StateFieldInfo:
    return StateFieldInfo(
        name="playTrigger",
        label="Play",
        tooltip="Increment to trigger playback.",
        show_on_node=True,
        access="rw",
        access_str="rw",
        required=True,
        ui_control="button",
        ui_language=None,
        value_schema=integer_schema(),
    )


def _invalid_button_field() -> StateFieldInfo:
    return StateFieldInfo(
        name="badTrigger",
        label="Bad",
        tooltip="Wrong schema for button.",
        show_on_node=True,
        access="rw",
        access_str="rw",
        required=True,
        ui_control="button",
        ui_language=None,
        value_schema=string_schema(),
    )


def _dial_field() -> StateFieldInfo:
    return StateFieldInfo(
        name="pan",
        label="Pan",
        tooltip="Circular pan control.",
        show_on_node=True,
        access="rw",
        access_str="rw",
        required=True,
        ui_control="dial",
        ui_language=None,
        value_schema=number_schema(default=0.0, minimum=-1.0, maximum=1.0),
    )


def _wrapline_field() -> StateFieldInfo:
    return StateFieldInfo(
        name="expr",
        label="Expr",
        tooltip="Single-line python expression.",
        show_on_node=True,
        access="rw",
        access_str="rw",
        required=True,
        ui_control="wrapline[python]",
        ui_language="python",
        value_schema=string_schema(default="x"),
    )


class _FakePropertyNode:
    def __init__(self, field: F8StateSpec) -> None:
        self._field = field

    def effective_state_fields(self) -> list[F8StateSpec]:
        return [self._field]


class _FakePropertyGraph:
    def __init__(self, node: "_FakeGraphPropertyNode") -> None:
        self.node = node

    def get_node_by_id(self, node_id: str) -> "_FakeGraphPropertyNode | None":
        if str(node_id or "") == self.node.id:
            return self.node
        return None


class _FakeGraphPropertyNode(_FakePropertyNode):
    def __init__(self, field: F8StateSpec, *, node_id: str = "nodeA", code: str = "") -> None:
        super().__init__(field)
        self.id = node_id
        self.graph = _FakePropertyGraph(self)
        self._props: dict[str, Any] = {str(field.name or ""): code}
        self.writes: list[tuple[str, Any, bool]] = []

    def name(self) -> str:
        return "nodeA"

    def get_property(self, name: str) -> Any:
        key = str(name or "")
        if key not in self._props:
            raise KeyError(key)
        return self._props[key]

    def set_property(self, name: str, value: Any, *, push_undo: bool = True) -> None:
        key = str(name or "")
        if key not in self._props:
            raise KeyError(key)
        self._props[key] = value
        self.writes.append((key, value, push_undo))


class _EnsureStateBackendNode:
    def __init__(self, fields: list[F8StateSpec], props: dict[str, Any] | None = None) -> None:
        self._fields = list(fields)
        self._props = dict(props or {})
        self.spec = None
        self.id = "nodeA"

    def effective_state_fields(self) -> list[F8StateSpec]:
        return list(self._fields)

    def get_property(self, name: str) -> Any:
        return self._props.get(str(name), None)

    def set_property(self, name: str, value: Any, *, push_undo: bool = True) -> None:
        del push_undo
        self._props[str(name)] = value


class _EnsureStateNodeItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, fields: list[F8StateSpec], *, props: dict[str, Any] | None = None) -> None:
        super().__init__(0.0, 0.0, 10.0, 10.0)
        self.id = "nodeA"
        self.name = "nodeA"
        self._backend = _EnsureStateBackendNode(fields, props=props)
        self._state_inline_proxies: dict[str, QtWidgets.QGraphicsProxyWidget] = {}
        self._state_inline_controls: dict[str, QtWidgets.QWidget] = {}
        self._state_inline_bindings: dict[str, Any] = {}
        self._state_inline_updaters: dict[str, Any] = {}
        self._state_inline_toggles: dict[str, Any] = {}
        self._state_inline_headers: dict[str, QtWidgets.QWidget] = {}
        self._state_inline_bodies: dict[str, QtWidgets.QWidget] = {}
        self._state_inline_expanded: dict[str, bool] = {}
        self._state_inline_option_pools: dict[str, str] = {}
        self._state_inline_ctrl_serial: dict[str, str] = {}
        self._tooltip_filters: list[Any] = []

    def _ensure_graph_property_hook(self) -> None:
        return

    def _backend_node(self) -> _EnsureStateBackendNode:
        return self._backend

    def _schema_enum_items(self, schema: Any) -> list[str]:
        del schema
        return []

    def _schema_numeric_range(self, schema: Any) -> tuple[float | None, float | None]:
        del schema
        return None, None

    def _is_state_inline_input_connected(self, field_name: str) -> bool:
        del field_name
        return False

    def _build_state_inline_control(
        self,
        info: StateFieldInfo,
        *,
        widget_parent: QtWidgets.QWidget | None = None,
    ) -> QtWidgets.QWidget:
        del widget_parent
        return QtWidgets.QLabel(info.label or info.name)

    def _toggle_state_inline_section(self, name: str, expanded: bool) -> None:
        self._state_inline_expanded[name] = bool(expanded)

    def _select_node_from_embedded_widget(self) -> None:
        return

    def _invalidate_layout_metrics(self) -> None:
        return

    def _prepare_layout_metrics(self) -> None:
        return

    def sync_proxy_mode(self, *, force: bool = False) -> None:
        del force
        return


class _EnsureRealStateNodeItem(_EnsureStateNodeItem):
    def _build_state_inline_control(
        self,
        info: StateFieldInfo,
        *,
        widget_parent: QtWidgets.QWidget | None = None,
    ) -> QtWidgets.QWidget:
        return build_state_inline_control(self, info, widget_parent=widget_parent)


def _mouse_event(
    event_type: QtCore.QEvent.Type,
    pos: QtCore.QPointF,
    *,
    button: QtCore.Qt.MouseButton,
    buttons: QtCore.Qt.MouseButton,
) -> QtGui.QMouseEvent:
    return QtGui.QMouseEvent(
        event_type,
        pos,
        pos,
        pos,
        button,
        buttons,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )


def _dial_pos(widget: QtWidgets.QWidget, fraction: float) -> QtCore.QPointF:
    rect = QtCore.QRectF(widget.rect()).adjusted(2.0, 2.0, -2.0, -2.0)
    center = rect.center()
    radius = min(rect.width(), rect.height()) / 2.0
    theta = (float(fraction) * 2.0 * math.pi) - (math.pi / 2.0)
    return QtCore.QPointF(center.x() + math.cos(theta) * radius, center.y() + math.sin(theta) * radius)


def test_build_state_inline_control_code_uses_push_button_and_style() -> None:
    _ensure_app()
    node_item = _FakeNodeItem(code_value="a\nb")
    control = build_state_inline_control(node_item, _code_field())

    assert isinstance(control, F8CodeButtonEditor)
    style = str(control.styleSheet() or "")
    assert "border:" in style
    assert "text-align: center" in style


def test_build_state_inline_control_code_installs_tooltip_filter_and_multiline_tooltip() -> None:
    _ensure_app()
    node_item = _FakeNodeItem(code_value="a\nb")
    control = build_state_inline_control(node_item, _code_field())
    assert isinstance(control, F8CodeButtonEditor)

    assert len(node_item._tooltip_filters) == 1
    tooltip_filter = node_item._tooltip_filters[0]
    assert isinstance(tooltip_filter, F8ForceGlobalToolTipFilter)
    assert tooltip_filter.parent() is control

    assert "2 lines" in str(control.toolTip() or "")

    updater = node_item._state_inline_updaters.get("code")
    assert callable(updater)
    updater("x\ny\nz")
    assert "3 lines" in str(control.toolTip() or "")


def test_build_state_inline_control_wave_preview_restores_widget() -> None:
    _ensure_app()
    node_item = _FakeNodeItem(code_value="")
    node_item._backend = _FakeBackendNode(
        {
            "express": "0.5 + 0.5 * cos(t)",
            "preview": [[0.0, 0.0], [0.1, 0.5], [0.2, 1.0]],
            "minValue": -1.0,
            "maxValue": 1.0,
            "maxT": 10.0,
        }
    )

    control = build_state_inline_control(node_item, _wave_preview_field())

    assert isinstance(control, WavePreviewControl)


def test_build_state_inline_control_wave_heatmap_restores_widget() -> None:
    _ensure_app()
    node_item = _FakeNodeItem(code_value="")
    node_item._backend = _FakeBackendNode(
        {
            "heatmap": [0.0, 0.5, 1.0],
            "maxT": 12.0,
        }
    )

    control = build_state_inline_control(node_item, _wave_heatmap_field())

    assert isinstance(control, WaveHeatmapControl)


def test_build_state_inline_control_selected_axis_uses_option_pool() -> None:
    _ensure_app()
    node_item = _FakeNodeItem(code_value="")
    node_item._backend = _FakeBackendNode(
        {
            "selectedAxis": "TopLevel",
            "allAxes": ["TopLevel", "L1", "R1"],
        }
    )

    control = build_state_inline_control(node_item, _selected_axis_field())

    assert isinstance(control, F8OptionCombo)


def test_build_state_inline_control_button_increments_integer_value() -> None:
    _ensure_app()
    node_item = _FakeNodeItem(code_value="")
    node_item._backend = _FakeBackendNode({"playTrigger": 0})

    control = build_state_inline_control(node_item, _button_field())

    assert isinstance(control, F8IncrementButtonEditor)
    assert control.text() == "Play"
    control.click()
    assert node_item._backend.get_property("playTrigger") == 1
    control.click()
    assert node_item._backend.get_property("playTrigger") == 2


def test_build_state_inline_control_button_disables_non_numeric_schema() -> None:
    _ensure_app()
    node_item = _FakeNodeItem(code_value="")
    node_item._backend = _FakeBackendNode({"badTrigger": "abc"})

    control = build_state_inline_control(node_item, _invalid_button_field())

    assert isinstance(control, F8IncrementButtonEditor)
    assert not control.isEnabled()
    assert "integer or number" in str(control.toolTip() or "")


def test_build_state_inline_control_dial_updates_backend_value() -> None:
    _ensure_app()
    node_item = _FakeNodeItem(code_value="")
    node_item._backend = _FakeBackendNode({"pan": 0.0})

    control = build_state_inline_control(node_item, _dial_field())

    assert isinstance(control, F8Dial)
    control.resize(96, 96)
    target = _dial_pos(control, 0.625)
    control.mousePressEvent(
        _mouse_event(
            QtCore.QEvent.Type.MouseButtonPress,
            target,
            button=QtCore.Qt.MouseButton.LeftButton,
            buttons=QtCore.Qt.MouseButton.LeftButton,
        )
    )
    control.mouseReleaseEvent(
        _mouse_event(
            QtCore.QEvent.Type.MouseButtonRelease,
            target,
            button=QtCore.Qt.MouseButton.LeftButton,
            buttons=QtCore.Qt.MouseButton.NoButton,
        )
    )

    assert float(node_item._backend.get_property("pan")) > 0.2


def test_build_state_inline_control_dial_installs_global_tooltip_filter() -> None:
    _ensure_app()
    node_item = _FakeNodeItem(code_value="")
    node_item._backend = _FakeBackendNode({"pan": 0.0})

    control = build_state_inline_control(node_item, _dial_field())

    assert isinstance(control, F8Dial)
    assert len(node_item._tooltip_filters) == 1
    tooltip_filter = node_item._tooltip_filters[0]
    assert isinstance(tooltip_filter, F8ForceGlobalToolTipFilter)
    assert tooltip_filter.parent() is control


def test_build_state_inline_control_dial_noloop_sets_loop_mode() -> None:
    _ensure_app()
    field = StateFieldInfo(
        name="pan",
        label="Pan",
        tooltip="Circular pan control.",
        show_on_node=True,
        access="rw",
        access_str="rw",
        required=True,
        ui_control="dial[noloop]",
        ui_language=None,
        value_schema=number_schema(default=0.0, minimum=-1.0, maximum=1.0),
    )
    node_item = _FakeNodeItem(code_value="")
    node_item._backend = _FakeBackendNode({"pan": 0.0})

    control = build_state_inline_control(node_item, field)

    assert isinstance(control, F8Dial)
    assert control.loop() is False


def test_build_state_inline_control_wrapline_python_skips_editor_assist_lookup(monkeypatch) -> None:
    _ensure_app()
    node_item = _FakeNodeItem(code_value="")
    node_item._backend = _FakeBackendNode({"expr": "x + 1"})

    def _unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("wrapline control should not request editor assist context")

    monkeypatch.setattr(
        "f8pystudio.nodegraph.items.state_inline_controls.editor_assist_context_for_field",
        _unexpected,
    )

    control = build_state_inline_control(node_item, _wrapline_field())

    assert isinstance(control, F8WrapLineEditor)


def test_build_state_panel_control_button_uses_field_label_and_increments() -> None:
    _ensure_app()
    field = F8StateSpec(
        name="playTrigger",
        label="Play",
        valueSchema=integer_schema(),
        access=F8StateAccess.rw,
        uiControl="button",
    )
    node = _FakePropertyNode(field)
    widget = build_state_panel_control(
        node=node,
        prop_name="playTrigger",
        widget_type=1,
        widget_factory=NodePropertyWidgetFactory(),
    )

    assert isinstance(widget, F8IncrementButtonEditor)
    assert widget.text() == "Play"
    seen: list[object] = []
    widget.value_changed.connect(lambda _name, value: seen.append(value))  # type: ignore[attr-defined]
    widget.click()
    widget.click()
    assert seen == [1, 2]


def test_build_state_panel_control_dial_uses_dial_editor() -> None:
    _ensure_app()
    field = F8StateSpec(
        name="pan",
        label="Pan",
        valueSchema=number_schema(default=0.0, minimum=-1.0, maximum=1.0),
        access=F8StateAccess.rw,
        uiControl="dial",
    )
    node = _FakePropertyNode(field)
    widget = build_state_panel_control(
        node=node,
        prop_name="pan",
        widget_type=1,
        widget_factory=NodePropertyWidgetFactory(),
    )

    assert isinstance(widget, F8DialEditor)


def test_build_state_panel_control_range_slider_uses_one_two_value_editor() -> None:
    _ensure_app()
    field = F8StateSpec(
        name="outputRange",
        label="Output Range",
        valueSchema=array_schema(
            items=number_schema(minimum=0.0, maximum=1.0),
            default=[0.2, 0.8],
        ),
        access=F8StateAccess.rw,
        uiControl="range_slider",
    )
    node = _FakePropertyNode(field)
    widget = build_state_panel_control(
        node=node,
        prop_name="outputRange",
        widget_type=1,
        widget_factory=NodePropertyWidgetFactory(),
    )

    assert isinstance(widget, F8RangeBarEditor)
    range_bar = widget.findChild(F8RangeBar)
    assert range_bar is not None
    assert range_bar.layout().count() == 2


def test_build_state_panel_control_dial_noloop_sets_loop_mode() -> None:
    _ensure_app()
    field = F8StateSpec(
        name="pan",
        label="Pan",
        valueSchema=number_schema(default=0.0, minimum=-1.0, maximum=1.0),
        access=F8StateAccess.rw,
        uiControl="dial[noloop]",
    )
    node = _FakePropertyNode(field)
    widget = build_state_panel_control(
        node=node,
        prop_name="pan",
        widget_type=1,
        widget_factory=NodePropertyWidgetFactory(),
    )

    assert isinstance(widget, F8DialEditor)
    dial = widget.findChild(F8Dial)
    assert dial is not None
    assert dial.loop() is False


def test_build_state_panel_control_dial_disables_non_numeric_schema() -> None:
    _ensure_app()
    field = F8StateSpec(
        name="badDial",
        label="Bad Dial",
        valueSchema=string_schema(default="oops"),
        access=F8StateAccess.rw,
        uiControl="dial",
    )
    node = _FakePropertyNode(field)
    widget = build_state_panel_control(
        node=node,
        prop_name="badDial",
        widget_type=1,
        widget_factory=NodePropertyWidgetFactory(),
    )

    assert isinstance(widget, F8DialEditor)
    dial = widget.findChild(F8Dial)
    assert dial is not None
    assert not dial.isEnabled()
    assert "integer or number" in str(dial.toolTip() or "")


def test_build_state_panel_control_code_save_persists_by_graph_node_id_after_widget_deleted() -> None:
    _ensure_app()
    field = F8StateSpec(
        name="code",
        label="Code",
        valueSchema=string_schema(default="print('old')\n"),
        access=F8StateAccess.rw,
        uiControl="code[python]",
    )
    node = _FakeGraphPropertyNode(field, code="print('old')\n")
    widget = build_state_panel_control(
        node=node,
        prop_name="code",
        widget_type=1,
        widget_factory=NodePropertyWidgetFactory(),
    )

    assert isinstance(widget, F8CodeButtonEditor)
    persisted_value_setter = widget._persisted_value_setter
    assert persisted_value_setter is not None
    widget.deleteLater()
    QtWidgets.QApplication.processEvents()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)

    assert persisted_value_setter("print('updated')\n") is True
    assert node.get_property("code") == "print('updated')\n"
    assert node.writes == [("code", "print('updated')\n", True)]


def test_build_state_panel_control_wrapline_python_skips_editor_assist_lookup(monkeypatch) -> None:
    _ensure_app()
    field = F8StateSpec(
        name="expr",
        label="Expr",
        valueSchema=string_schema(default="x"),
        access=F8StateAccess.rw,
        uiControl="wrapline[python]",
    )
    node = _FakePropertyNode(field)

    def _unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("wrapline control should not request editor assist context")

    monkeypatch.setattr(
        "f8pystudio.ui.support.state_panel_controls.editor_assist_context_for_field",
        _unexpected,
    )

    widget = build_state_panel_control(
        node=node,
        prop_name="expr",
        widget_type=1,
        widget_factory=NodePropertyWidgetFactory(),
    )

    assert isinstance(widget, F8WrapLineEditor)


def test_ensure_state_inline_controls_disposes_detached_widget_without_reparent_flash(monkeypatch) -> None:
    _ensure_app()
    first = F8StateSpec(
        name="code",
        label="Code",
        access=F8StateAccess.rw,
        uiControl="code",
        showOnNode=True,
        valueSchema=string_schema(),
    )
    second = F8StateSpec(
        name="preview",
        label="Preview",
        access=F8StateAccess.ro,
        uiControl="text",
        showOnNode=True,
        valueSchema=string_schema(),
    )
    node_item = _EnsureStateNodeItem([first])

    ensure_state_inline_controls(node_item)
    old_widget = node_item._state_inline_proxies["code"].widget()
    assert old_widget is not None

    seen: list[tuple[QtWidgets.QWidget, str]] = []

    def _record(widget: QtWidgets.QWidget | None, *, context: str) -> None:
        if widget is not None:
            seen.append((widget, context))

    monkeypatch.setattr(
        "f8pystudio.nodegraph.items.state_inline_controls.dispose_detached_proxy_widget",
        _record,
    )

    node_item._backend = _EnsureStateBackendNode([second])
    ensure_state_inline_controls(node_item)

    assert seen == [(old_widget, "inline-state-remove:code")]


def test_ensure_state_inline_controls_reorders_renamed_field_to_match_spec_order() -> None:
    _ensure_app()
    first = F8StateSpec(name="first", access=F8StateAccess.rw, showOnNode=True, valueSchema=string_schema())
    second = F8StateSpec(name="second", access=F8StateAccess.rw, showOnNode=True, valueSchema=string_schema())
    renamed = F8StateSpec(name="renamed", access=F8StateAccess.rw, showOnNode=True, valueSchema=string_schema())
    node_item = _EnsureStateNodeItem([first, second])

    ensure_state_inline_controls(node_item)
    assert list(node_item._state_inline_proxies.keys()) == ["first", "second"]

    node_item._backend._fields = [renamed, second]
    ensure_state_inline_controls(node_item)

    assert list(node_item._state_inline_proxies.keys()) == ["renamed", "second"]


def test_ensure_state_inline_controls_refreshes_existing_button_label_and_tooltip() -> None:
    _ensure_app()
    field = F8StateSpec(
        name="playTrigger",
        label="Play",
        description="Increment to trigger playback.",
        access=F8StateAccess.rw,
        uiControl="button",
        showOnNode=True,
        valueSchema=integer_schema(),
    )
    node_item = _EnsureRealStateNodeItem([field], props={"playTrigger": 0})

    ensure_state_inline_controls(node_item)

    control = node_item._state_inline_controls["playTrigger"]
    assert isinstance(control, F8IncrementButtonEditor)
    assert control.text() == "Play"
    assert "Increment to trigger playback." in str(control.toolTip() or "")

    updated_field = copy_model(
        field,
        update={
            "label": "Start",
            "description": "Increment to start playback.",
        },
    )
    node_item._backend._fields = [updated_field]

    ensure_state_inline_controls(node_item)

    updated_control = node_item._state_inline_controls["playTrigger"]
    assert updated_control is control
    assert isinstance(updated_control, F8IncrementButtonEditor)
    assert updated_control.text() == "Start"
    assert "Increment to start playback." in str(updated_control.toolTip() or "")


def test_state_inline_control_serial_changes_when_numeric_range_changes() -> None:
    node_item = _SerialNodeItem()
    info_a = StateFieldInfo(
        name="pan",
        label="Pan",
        tooltip="Circular pan control.",
        show_on_node=True,
        access="rw",
        access_str="rw",
        required=True,
        ui_control="dial",
        ui_language="",
        value_schema=number_schema(default=0.0, minimum=-1.0, maximum=1.0),
    )
    info_b = StateFieldInfo(
        name="pan",
        label="Pan",
        tooltip="Circular pan control.",
        show_on_node=True,
        access="rw",
        access_str="rw",
        required=True,
        ui_control="dial",
        ui_language="",
        value_schema=number_schema(default=0.0, minimum=-2.0, maximum=2.0),
    )

    serial_a = state_inline_control_serial(node_item, info_a)
    serial_b = state_inline_control_serial(node_item, info_b)

    assert serial_a != serial_b


def test_option_combo_read_only_toggle_does_not_call_qlineedit_text_interaction_flags() -> None:
    _ensure_app()
    control = F8OptionCombo()
    control.set_options(["TopLevel", "L1", "R1"])
    control.set_value("TopLevel")

    control.set_read_only(True)

    assert control.isEditable()
    line_edit = control.lineEdit()
    assert line_edit is not None
    assert line_edit.isReadOnly()

    control.set_read_only(False)

    assert not control.isEditable()


def test_sync_state_inline_controls_from_graph_property_updates_wave_heatmap() -> None:
    _ensure_app()
    node_item = _FakeNodeItem(code_value="")
    seen: list[Any] = []

    def _record_heatmap(value: Any) -> None:
        seen.append(value)

    node_item._state_inline_updaters["heatmap"] = _record_heatmap
    sync_state_inline_controls_from_graph_property(node_item, node_item._backend, "heatmap", [0.0, 1.0, 0.0])

    assert seen == [[0.0, 1.0, 0.0]]


def test_build_state_inline_control_wave_pattern_restores_widget() -> None:
    _ensure_app()
    node_item = _FakeNodeItem(code_value="")
    node_item._backend = _FakeBackendNode(
        {
            "points": [[0.0, 0.0], [10.0, 0.0]],
            "preview": [[0.0, 0.0], [1.0, 0.1], [2.0, 0.0]],
            "minValue": 0.0,
            "maxValue": 1.0,
            "maxT": 10.0,
        }
    )

    control = build_state_inline_control(node_item, _wave_pattern_field())

    assert isinstance(control, WavePatternEditorControl)


def test_sync_state_inline_controls_from_graph_property_refreshes_wave_preview_from_bounds_change() -> None:
    _ensure_app()
    node_item = _FakeNodeItem(code_value="")
    node_item._backend = _FakeBackendNode(
        {
            "express": "sin(t)",
            "preview": [[0.0, 0.0], [0.1, 1.0]],
            "minValue": -1.0,
            "maxValue": 1.0,
            "maxT": 4.0,
        }
    )
    seen: list[Any] = []

    def _record_preview(value: Any) -> None:
        seen.append(value)

    node_item._state_inline_updaters["preview"] = _record_preview

    sync_state_inline_controls_from_graph_property(node_item, node_item._backend, "maxT", 8.0)

    assert seen == [[[0.0, 0.0], [0.1, 1.0]]]


def test_sync_state_inline_controls_from_graph_property_refreshes_wave_pattern_from_preview_dependency() -> None:
    _ensure_app()
    node_item = _FakeNodeItem(code_value="")
    node_item._backend = _FakeBackendNode(
        {
            "points": [[0.0, 0.0], [10.0, 0.0]],
            "preview": [[0.0, 0.0], [0.1, 1.0]],
            "minValue": 0.0,
            "maxValue": 1.0,
            "maxT": 10.0,
        }
    )
    seen: list[Any] = []

    def _record_points(value: Any) -> None:
        seen.append(value)

    node_item._state_inline_updaters["points"] = _record_points

    sync_state_inline_controls_from_graph_property(node_item, node_item._backend, "preview", [[0.0, 0.0], [0.2, 0.9]])

    assert seen == [[[0.0, 0.0], [10.0, 0.0]]]


def test_wave_pattern_editor_preserves_hidden_points_when_max_t_shrinks() -> None:
    _ensure_app()
    node_item = _FakeNodeItem(code_value="")
    node_item._backend = _FakeBackendNode(
        {
            "points": [[1.0, 0.1], [6.0, 0.6], [12.0, 1.2]],
            "preview": [[0.0, 0.1], [2.5, 0.1]],
            "minValue": 0.0,
            "maxValue": 2.0,
            "maxT": 12.0,
        }
    )

    control = build_state_inline_control(node_item, _wave_pattern_field())
    assert isinstance(control, WavePatternEditorControl)

    node_item._backend.set_property("maxT", 5.0)
    sync_state_inline_controls_from_graph_property(node_item, node_item._backend, "maxT", 5.0)

    assert node_item._backend.get_property("points") == [[1.0, 0.1], [6.0, 0.6], [12.0, 1.2]]


def test_wave_pattern_editor_add_move_delete_updates_backend_points() -> None:
    _ensure_app()
    node_item = _FakeNodeItem(code_value="")
    node_item._backend = _FakeBackendNode(
        {
            "points": [[0.0, 0.0], [10.0, 0.0]],
            "preview": [[0.0, 0.0], [5.0, 0.5], [9.0, 0.0]],
            "minValue": 0.0,
            "maxValue": 1.0,
            "maxT": 10.0,
        }
    )

    control = build_state_inline_control(node_item, _wave_pattern_field())
    assert isinstance(control, WavePatternEditorControl)
    control.resize(240, 84)

    add_pos = QtCore.QPointF(120.0, 18.0)
    control.mousePressEvent(
        _mouse_event(
            QtCore.QEvent.Type.MouseButtonPress,
            add_pos,
            button=QtCore.Qt.MouseButton.LeftButton,
            buttons=QtCore.Qt.MouseButton.LeftButton,
        )
    )
    control.mouseReleaseEvent(
        _mouse_event(
            QtCore.QEvent.Type.MouseButtonRelease,
            add_pos,
            button=QtCore.Qt.MouseButton.LeftButton,
            buttons=QtCore.Qt.MouseButton.NoButton,
        )
    )

    points_after_add = node_item._backend.get_property("points")
    assert len(points_after_add) == 3

    rect = graph_draw_rect(control.rect())
    move_from = point_to_widget_pos(
        points_after_add[1][0], points_after_add[1][1], rect=rect, max_t=10.0, y_range=(0.0, 1.0)
    )
    move_to = QtCore.QPointF(move_from.x() + 20.0, move_from.y() + 18.0)
    control.mousePressEvent(
        _mouse_event(
            QtCore.QEvent.Type.MouseButtonPress,
            move_from,
            button=QtCore.Qt.MouseButton.LeftButton,
            buttons=QtCore.Qt.MouseButton.LeftButton,
        )
    )
    control.mouseMoveEvent(
        _mouse_event(
            QtCore.QEvent.Type.MouseMove,
            move_to,
            button=QtCore.Qt.MouseButton.NoButton,
            buttons=QtCore.Qt.MouseButton.LeftButton,
        )
    )
    control.mouseReleaseEvent(
        _mouse_event(
            QtCore.QEvent.Type.MouseButtonRelease,
            move_to,
            button=QtCore.Qt.MouseButton.LeftButton,
            buttons=QtCore.Qt.MouseButton.NoButton,
        )
    )

    points_after_move = node_item._backend.get_property("points")
    assert points_after_move[1][0] > points_after_add[1][0]
    assert points_after_move[1][1] < points_after_add[1][1]

    moved_pos = point_to_widget_pos(
        points_after_move[1][0], points_after_move[1][1], rect=rect, max_t=10.0, y_range=(0.0, 1.0)
    )
    control.mousePressEvent(
        _mouse_event(
            QtCore.QEvent.Type.MouseButtonPress,
            moved_pos,
            button=QtCore.Qt.MouseButton.RightButton,
            buttons=QtCore.Qt.MouseButton.RightButton,
        )
    )

    points_after_delete = node_item._backend.get_property("points")
    assert len(points_after_delete) == 2


def test_wave_preview_auto_zoom_when_min_gte_max() -> None:
    y_range = WavePreviewControl._coerce_preview_y_range(0.0, 0.0)
    assert y_range is None
