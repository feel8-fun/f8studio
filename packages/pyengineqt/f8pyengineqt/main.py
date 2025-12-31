from __future__ import annotations

from typing import Iterable

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSpec,
    F8StateFieldAccess,
    F8StateSpec,
    number_schema,
    string_schema,
)

from f8pyengineqt.editor.operator_graph_editor import OperatorGraphEditor
from f8pyengineqt.graph.operator_graph import OperatorGraph
from f8pyengineqt.renderers.renderer_registry import OperatorRendererRegistry
from f8pyengineqt.operators.operator_registry import OperatorSpecRegistry


def _demo_specs() -> Iterable[F8OperatorSpec]:
    """Small built-in palette so the app can launch without external assets."""
    yield F8OperatorSpec(
        operatorClass="feel8.sample.start",
        version="0.0.1",
        label="Start",
        description="Entry trigger for demo graphs.",
        execOutPorts=["exec"],
    )
    yield F8OperatorSpec(
        operatorClass="feel8.sample.constant",
        version="0.0.1",
        label="Constant",
        description="Emits a configured numeric value.",
        execInPorts=["exec"],
        execOutPorts=["exec"],
        dataOutPorts=[F8DataPortSpec(name="value", valueSchema=number_schema(), description="constant output")],
        states=[
            F8StateSpec(
                name="value",
                label="Value",
                valueSchema=number_schema(default=1.0, minimum=-1000, maximum=1000),
                access=F8StateFieldAccess.rw,
            )
        ],
    )
    yield F8OperatorSpec(
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
                access=F8StateFieldAccess.rw,
            )
        ],
    )


def _seed_graph(view: OperatorGraphEditor) -> None:
    """Populate the graph with demo nodes and links."""
    start = view.spawn_instance("feel8.sample.start", pos=(-420, 0))
    const_a = view.spawn_instance("feel8.sample.constant", pos=(-180, -120), instance_id="const_a")
    const_b = view.spawn_instance("feel8.sample.constant", pos=(-180, 120), instance_id="const_b")
    add = view.spawn_instance("feel8.sample.add", pos=(140, 0))
    logger = view.spawn_instance("feel8.sample.log", pos=(420, 0))

    # Example: tweak spec/state per instance.
    const_a.state["value"] = 3.0
    const_b.state["value"] = 7.0
    logger.state["label"] = "Sum"

    graph = view.graph
    graph.connect_exec(start.id, "exec", const_a.id, "exec")
    graph.connect_exec(const_a.id, "exec", const_b.id, "exec")
    graph.connect_exec(const_b.id, "exec", add.id, "exec")

    graph.connect_data(const_a.id, "value", add.id, "a")
    graph.connect_data(const_b.id, "value", add.id, "b")
    graph.connect_data(add.id, "sum", logger.id, "value")


def main() -> None:
    from qtpy import QtWidgets  # type: ignore[import-not-found]
    from NodeGraphQt import NodesPaletteWidget

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    OperatorSpecRegistry().instance().register_many(_demo_specs(), overwrite=True)
    
    graph = OperatorGraph()

    view = OperatorGraphEditor(
        graph=graph,
    )
    _seed_graph(view)

    graph_widget = view.widget()
    palette = NodesPaletteWidget(node_graph=view.node_graph)

    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Feel8 Graph")
    splitter = QtWidgets.QSplitter()
    splitter.addWidget(palette)
    splitter.addWidget(graph_widget)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([260, 940])
    window.setCentralWidget(splitter)
    window.resize(1200, 720)
    window.show()

    if hasattr(app, "exec_"):
        app.exec_()
    else:
        app.exec()


if __name__ == "__main__":
    main()
