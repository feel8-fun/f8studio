from f8engine import Access, OperatorGraph, OperatorGraphEditor, OperatorRegistry, OperatorSpec, StateField, Type


def build_demo_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    registry.register(
        OperatorSpec(
            schemaVersion="f8operator/1",
            operatorClass="fun.feel8.ops.tick",
            version="0.1.0",
            label="Tick",
            tags=["timer", "entry"],
            description="Emit exec pulses",
            execInPorts=[],
            execOutPorts=["tick"],
            allowAddExecInPorts=False,
            allowAddExecOutPorts=False,
            states=[
                StateField(
                    name="interval",
                    label="Interval (s)",
                    type=Type.float,
                    default=0.5,
                    access=Access.rw,
                )
            ],
            allowAddStates=False,
            commands=[],
            allowAddCommands=False,
            dataInPorts=[],
            dataOutPorts=[],
            allowAddDataInPorts=False,
            allowAddDataOutPorts=False,
        )
    )
    registry.register(
        OperatorSpec(
            schemaVersion="f8operator/1",
            operatorClass="fun.feel8.ops.log",
            version="0.1.0",
            tags=["debug", "output"],
            label="Log",
            description="Print incoming payloads",
            execInPorts=["in"],
            execOutPorts=["next"],
            allowAddExecInPorts=False,
            allowAddExecOutPorts=False,
            dataInPorts=[
                {"name": "message", "type": Type.string, "description": "Text to log"},
            ],
            dataOutPorts=[],
            allowAddDataInPorts=False,
            allowAddDataOutPorts=False,
            states=[
                StateField(
                    name="level",
                    label="Level",
                    type=Type.string,
                    default="info",
                    access=Access.rw,
                )
            ],
            allowAddStates=False,
            commands=[],
            allowAddCommands=False,
        )
    )
    return registry


if __name__ == "__main__":
    registry = build_demo_registry()
    graph = OperatorGraph()
    
    OperatorGraphEditor(operatorCls_registry=registry, graph=graph).run()
