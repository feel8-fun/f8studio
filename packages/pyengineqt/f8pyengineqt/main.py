from __future__ import annotations

from typing import Iterable

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSpec,
    F8PrimitiveType,
    F8PrimitiveTypeEnum,
    F8StateFieldAccess,
    F8StateSpec,
)

from f8pyengineqt.graph_view import OperatorGraphView
from f8pyengineqt.operator_graph import OperatorGraph
from f8pyengineqt.renderer import OperatorRendererRegistry
from f8pyengineqt.spec_registry import OperatorSpecRegistry


def _number_schema(
    *,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> F8PrimitiveType:
    """Convenience factory for numeric schemas used across demo specs."""
    return F8PrimitiveType(
            type=F8PrimitiveTypeEnum.number,
            default=default,
            minimum=minimum,
            maximum=maximum,
        )
    


def _string_schema(*, default: str | None = None) -> F8PrimitiveType:
    return F8PrimitiveType(type=F8PrimitiveTypeEnum.string, default=default)


def _demo_specs() -> Iterable[F8OperatorSpec]:
    """Small built-in palette so the app can launch without external assets."""
    yield F8OperatorSpec(
        operatorClass="feel8.sample.start",
        version="0.0.1",
        label="Start",
        description="Entry trigger for demo graphs.",
        execOutPorts=["next"],
    )
    yield F8OperatorSpec(
        operatorClass="feel8.sample.constant",
        version="0.0.1",
        label="Constant",
        description="Emits a configured numeric value.",
        execInPorts=["in"],
        execOutPorts=["out"],
        dataOutPorts=[F8DataPortSpec(name="value", valueSchema=_number_schema(), description="constant output")],
        states=[
            F8StateSpec(
                name="value",
                label="Value",
                valueSchema=_number_schema(default=1.0, minimum=-1000, maximum=1000),
                access=F8StateFieldAccess.rw,
            )
        ],
    )
    yield F8OperatorSpec(
        operatorClass="feel8.sample.add",
        version="0.0.1",
        label="Add",
        description="Adds two inputs and forwards the result.",
        execInPorts=["enter"],
        execOutPorts=["next"],
        dataInPorts=[
            F8DataPortSpec(name="a", valueSchema=_number_schema(), description="lhs"),
            F8DataPortSpec(name="b", valueSchema=_number_schema(), description="rhs"),
        ],
        dataOutPorts=[F8DataPortSpec(name="sum", valueSchema=_number_schema(), description="a+b")],
    )
    yield F8OperatorSpec(
        operatorClass="feel8.sample.log",
        version="0.0.1",
        label="Log",
        description="Terminal node that inspects incoming data.",
        execInPorts=["enter"],
        dataInPorts=[
            F8DataPortSpec(name="value", valueSchema=_number_schema(), description="value to log", required=False),
        ],
        states=[
            F8StateSpec(
                name="label",
                label="Label",
                valueSchema=_string_schema(default="Log"),
                access=F8StateFieldAccess.rw,
            )
        ],
    )


def _seed_graph(view: OperatorGraphView) -> None:
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
    graph.connect_exec(start.id, "next", const_a.id, "in")
    graph.connect_exec(const_a.id, "out", const_b.id, "in")
    graph.connect_exec(const_b.id, "out", add.id, "enter")
    graph.connect_exec(add.id, "next", logger.id, "enter")

    graph.connect_data(const_a.id, "value", add.id, "a")
    graph.connect_data(const_b.id, "value", add.id, "b")
    graph.connect_data(add.id, "sum", logger.id, "value")

    view.rebuild()


def main() -> None:
    from qtpy import QtWidgets  # type: ignore[import-not-found]

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    spec_registry = OperatorSpecRegistry()
    spec_registry.register_many(_demo_specs(), overwrite=True)
    renderer_registry = OperatorRendererRegistry()
    graph = OperatorGraph()

    view = OperatorGraphView(
        spec_registry=spec_registry,
        renderer_registry=renderer_registry,
        graph=graph,
    )
    _seed_graph(view)

    widget = view.widget()
    if hasattr(widget, "resize"):
        widget.resize(1200, 720)
    view.show()

    if hasattr(app, "exec_"):
        app.exec_()
    else:
        app.exec()


if __name__ == "__main__":
    main()
