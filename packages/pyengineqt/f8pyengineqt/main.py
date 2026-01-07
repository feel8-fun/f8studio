from __future__ import annotations

import os
from typing import Iterable

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSpec,
    F8StateAccess,
    F8StateSpec,
    number_schema,
    string_schema,
    integer_schema,
    boolean_schema,
    array_schema,
    any_schema,
    complex_object_schema,
    operator_key,
)

from f8pyengineqt.editor.operator_graph_editor import OperatorGraphEditor
from f8pyengineqt.editor.service_graph_editor import ServiceGraphEditor
from f8pyengineqt.renderers.renderer_registry import OperatorRendererRegistry
from f8pyengineqt.services.builtin import EDITOR_SERVICE_CLASS, ENGINE_SERVICE_CLASS
from f8pyengineqt.services.service_operator_registry import ServiceOperatorSpecRegistry
from f8pyengineqt.services.discovery_loader import load_discovery_into_registries
from f8pyengineqt.renderers.generic import GenericNode


def _demo_specs() -> Iterable[F8OperatorSpec]:
    """Small built-in palette so the app can launch without external assets."""
    yield F8OperatorSpec(
        serviceClass=ENGINE_SERVICE_CLASS,
        operatorClass="feel8.sample.start",
        version="0.0.1",
        label="Start",
        description="Entry trigger for demo graphs.",
        execOutPorts=["exec"],
    )
    yield F8OperatorSpec(
        serviceClass=ENGINE_SERVICE_CLASS,
        operatorClass="feel8.sample.constant",
        version="0.0.1",
        label="Constant",
        description="Emits a configured numeric value.",
        execInPorts=["exec"],
        execOutPorts=["exec"],
        dataOutPorts=[F8DataPortSpec(name="value", valueSchema=number_schema(), description="constant output")],
        states=[
            F8StateSpec(
                name="value2",
                label="Value",
                valueSchema=number_schema(default=1.0, minimum=-1000, maximum=1000),
                access=F8StateAccess.rw,
            )
        ],
    )
    yield F8OperatorSpec(
        serviceClass=ENGINE_SERVICE_CLASS,
        operatorClass="feel8.sample.add",
        version="0.0.1",
        label="Add",
        description="Adds two inputs and forwards the result.",
        execInPorts=["exec"],
        execOutPorts=["exec"],
        dataInPorts=[
            F8DataPortSpec(name="a", valueSchema=number_schema(), description="lhs"),
            F8DataPortSpec(name="b", valueSchema=number_schema(), description="rhs"),
        ],
        dataOutPorts=[F8DataPortSpec(name="sum", valueSchema=number_schema(), description="a+b")],
    )
    yield F8OperatorSpec(
        serviceClass=ENGINE_SERVICE_CLASS,
        operatorClass="feel8.sample.log",
        version="0.0.1",
        label="Log",
        description="Terminal node that inspects incoming data.",
        tags=["ui"],
        execInPorts=[],
        dataInPorts=[
            F8DataPortSpec(name="value", valueSchema=number_schema(), description="value to log", required=False),
        ],
        states=[
            F8StateSpec(
                name="label",
                label="Label",
                valueSchema=string_schema(default="Log"),
                access=F8StateAccess.ro,
            ),
            F8StateSpec(
                name="bool",
                label="bool",
                valueSchema=boolean_schema(default=True),
                access=F8StateAccess.rw,
                showOnNode=True,
            ),
            F8StateSpec(
                name="count",
                label="Count",
                valueSchema=integer_schema(default=5, minimum=0, maximum=100),
                access=F8StateAccess.rw,
                showOnNode=True,
            ),
            F8StateSpec(
                name="options",
                label="Options",
                valueSchema=string_schema(enum=["Option A", "Option B", "Option C"]),
                access=F8StateAccess.ro,
            ),
            F8StateSpec(
                name="metadata",
                label="Metadata",
                valueSchema=complex_object_schema(
                    properties={
                        "author": string_schema(default="Unknown"),
                        "version": integer_schema(default=1, minimum=1),
                        "tags": array_schema(items=string_schema()),
                    }
                ),
                access=F8StateAccess.ro,
            ),
        ],
    )

    # Editor-only visualization nodes (run in the editor service, not in an engine).
    yield F8OperatorSpec(
        serviceClass=EDITOR_SERVICE_CLASS,
        operatorClass="feel8.editor.log",
        version="0.0.1",
        label="Editor Log",
        description="Receives data via cross edges and prints / displays it in the editor.",
        tags=["ui", "editor"],
        rendererClass="editor_log",
        dataInPorts=[F8DataPortSpec(name="in", valueSchema=any_schema(), description="data input")],
        states=[
            F8StateSpec(
                name="refreshMs",
                label="Refresh (ms)",
                valueSchema=integer_schema(default=200, minimum=16, maximum=5000),
                access=F8StateAccess.rw,
                showOnNode=True,
            ),
            F8StateSpec(
                name="print",
                label="Print",
                valueSchema=boolean_schema(default=True),
                access=F8StateAccess.rw,
                showOnNode=True,
            ),
        ],
    )
    yield F8OperatorSpec(
        serviceClass=EDITOR_SERVICE_CLASS,
        operatorClass="feel8.editor.oscilloscope",
        version="0.0.1",
        label="Oscilloscope",
        description="Receives numeric data via cross edges and visualizes it (placeholder UI).",
        tags=["ui", "editor"],
        rendererClass="ui",
        dataInPorts=[F8DataPortSpec(name="in", valueSchema=number_schema(), description="signal input")],
        states=[
            F8StateSpec(
                name="refreshMs",
                label="Refresh (ms)",
                valueSchema=integer_schema(default=50, minimum=16, maximum=5000),
                access=F8StateAccess.rw,
                showOnNode=True,
            ),
            F8StateSpec(
                name="window",
                label="Window",
                valueSchema=integer_schema(default=240, minimum=10, maximum=5000),
                access=F8StateAccess.rw,
                showOnNode=True,
            ),
        ],
    )


def _seed_graph(view: OperatorGraphEditor) -> None:
    """Populate the graph with demo nodes and links."""
    start = view.create_node(operator_key(ENGINE_SERVICE_CLASS, "feel8.sample.start"), pos=(-420, 0))
    const_a = view.create_node(operator_key(ENGINE_SERVICE_CLASS, "feel8.sample.constant"), pos=(-180, -120), name="const_a")
    const_b = view.create_node(operator_key(ENGINE_SERVICE_CLASS, "feel8.sample.constant"), pos=(-180, 120), name="const_b")
    add = view.create_node(operator_key(ENGINE_SERVICE_CLASS, "feel8.sample.add"), pos=(140, 0))
    logger = view.create_node(operator_key(ENGINE_SERVICE_CLASS, "feel8.sample.log"), pos=(420, 0))

    # const_a.set_property("value", 3.0, push_undo=False)
    # const_b.set_property("value", 7.0, push_undo=False)
    # logger.set_property("label", "Sum", push_undo=False)

    view.connect(start, kind="exec", out_port="exec", target=const_a, in_port="exec")
    view.connect(const_a, kind="exec", out_port="exec", target=const_b, in_port="exec")
    view.connect(const_b, kind="exec", out_port="exec", target=add, in_port="exec")

    view.connect(const_a, kind="data", out_port="value", target=add, in_port="a")
    view.connect(const_b, kind="data", out_port="value", target=add, in_port="b")
    view.connect(add, kind="data", out_port="sum", target=logger, in_port="value")


def _seed_operator_graph(node_graph: object) -> list[GenericNode]:
    """
    Populate a NodeGraphQt graph with demo operator nodes.

    Used by both the operator-graph demo and the service-graph demo.
    """
    create_kwargs = {"push_undo": False, "selected": False}
    try:
        start = node_graph.create_node(operator_key(ENGINE_SERVICE_CLASS, "feel8.sample.start"), pos=[-420.0, 0.0], **create_kwargs)
        const_a = node_graph.create_node(operator_key(ENGINE_SERVICE_CLASS, "feel8.sample.constant"), pos=[-180.0, -120.0], name="const_a", **create_kwargs)
        const_b = node_graph.create_node(operator_key(ENGINE_SERVICE_CLASS, "feel8.sample.constant"), pos=[-180.0, 120.0], name="const_b", **create_kwargs)
        add = node_graph.create_node(operator_key(ENGINE_SERVICE_CLASS, "feel8.sample.add"), pos=[140.0, 0.0], **create_kwargs)
        logger = node_graph.create_node(operator_key(ENGINE_SERVICE_CLASS, "feel8.sample.log"), pos=[420.0, 0.0], **create_kwargs)
    except Exception:
        return []

    nodes = [start, const_a, const_b, add, logger]
    if not all(isinstance(n, GenericNode) for n in nodes):
        return []

    def _connect(kind: str, out_node: GenericNode, out_port: str, in_node: GenericNode, in_port: str) -> None:
        src = out_node.port(kind, "out", out_port)
        dst = in_node.port(kind, "in", in_port)
        if src is None or dst is None:
            return
        try:
            src.connect_to(dst)
        except Exception:
            return

    _connect("exec", start, "exec", const_a, "exec")
    _connect("exec", const_a, "exec", const_b, "exec")
    _connect("exec", const_b, "exec", add, "exec")
    _connect("data", const_a, "value", add, "a")
    _connect("data", const_b, "value", add, "b")
    _connect("data", add, "sum", logger, "value")
    return [start, const_a, const_b, add, logger]


def main() -> None:
    from qtpy import QtWidgets  # type: ignore[import-not-found]
    from NodeGraphQt import NodesPaletteWidget

    from f8pyengineqt.editor.node_property_editor import NodePropertyEditorWidget

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # NodeGraphQt's PropertiesBinWidget spinbox widgets reference `self.NoButtons`
    # which is not available on PySide6 widget instances (only on the class).
    # Patch the python subclasses so `self.NoButtons` resolves via class lookup.
    # try:
    #     from Qt import QtWidgets as _QtWidgets  # NodeGraphQt uses Qt.py internally.
    #     from NodeGraphQt.custom_widgets.properties_bin import prop_widgets_base as _pwb

    #     _pwb.PropSpinBox.NoButtons = _QtWidgets.QAbstractSpinBox.NoButtons
    #     _pwb.PropDoubleSpinBox.NoButtons = _QtWidgets.QAbstractSpinBox.NoButtons
    # except Exception:
    #     pass

    loaded = bool(load_discovery_into_registries())
    if not loaded:
        # Fallback built-in demo specs (useful when running without a discovery folder).
        engine_specs = []
        editor_specs = []
        for spec in _demo_specs():
            svc = str(getattr(spec, "serviceClass", "") or "")
            if svc == EDITOR_SERVICE_CLASS:
                editor_specs.append(spec)
            else:
                engine_specs.append(spec)

        svc_ops = ServiceOperatorSpecRegistry.instance()
        svc_ops.register_many(ENGINE_SERVICE_CLASS, engine_specs, overwrite=True)
        svc_ops.register_many(EDITOR_SERVICE_CLASS, editor_specs, overwrite=True)

    OperatorRendererRegistry.instance()

    mode = (os.environ.get("F8_EDITOR_MODE") or "svc").strip().lower()

    if mode in ("service", "service-graph", "svc"):
        view = ServiceGraphEditor()
        graph_widget = view.widget()
        engine_node = view.node_graph.create_node(
            ENGINE_SERVICE_CLASS,
            name="Engine",
            pos=[-520.0, -260.0],
            push_undo=False,
            selected=False,
        )
        demo_nodes = _seed_operator_graph(view.node_graph)
        try:
            engine_node.wrap_operator_nodes(demo_nodes)
        except Exception:
            pass
        try:
            engine_node.set_text(f"serviceId={engine_node.id}")
        except Exception:
            pass
    else:
        view = OperatorGraphEditor()
        _seed_graph(view)
        graph_widget = view.widget()

    palette = NodesPaletteWidget(node_graph=view.node_graph)
    inspector = NodePropertyEditorWidget(parent=None)

    def _sync_inspector(*_args: object) -> None:
        inspector.set_selected_nodes(list(view.node_graph.selected_nodes()))

    view.node_graph.node_selection_changed.connect(_sync_inspector)
    view.node_graph.node_selected.connect(_sync_inspector)
    _sync_inspector()

    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Feel8 Graph")
    splitter = QtWidgets.QSplitter()
    splitter.addWidget(palette)
    splitter.addWidget(graph_widget)
    splitter.addWidget(inspector)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setStretchFactor(2, 0)
    splitter.setSizes([260, 700, 240])
    window.setCentralWidget(splitter)
    window.resize(1200, 720)
    window.show()

    if hasattr(app, "exec_"):
        app.exec_()
    else:
        app.exec()


if __name__ == "__main__":
    main()
