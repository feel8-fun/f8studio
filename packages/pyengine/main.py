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
            execInPorts=["exec"],
            execOutPorts=["exec"],
            allowAddExecInPorts=False,
            allowAddExecOutPorts=False,
            dataInPorts=[
                {"name": "message", "type": Type.string, "description": "Text to log"},
            ],
            dataOutPorts=[],
            allowAddDataInPorts=False,
            allowAddDataOutPorts=False,
            states=[],
            allowAddStates=False,
            commands=[],
            allowAddCommands=False,
        )
    )

    registry.register(
        OperatorSpec(
            schemaVersion="f8operator/1",
            operatorClass="fun.feel8.ops.sine",
            version="0.1.0",
            label="Sine Wave",
            tags=["math", "waveform"],
            description="Generate a sine wave signal",
            execInPorts=["exec"],
            execOutPorts=["exec"],
            allowAddExecInPorts=False,
            allowAddExecOutPorts=False,
            dataInPorts=[],
            dataOutPorts=[
                {"name": "value", "type": Type.float, "description": "Sine wave value"},
            ],
            allowAddDataInPorts=False,
            allowAddDataOutPorts=False,
            states=[
                StateField(
                    name="frequency",
                    label="Frequency (Hz)",
                    type=Type.float,
                    default=1,
                    access=Access.rw,
                ),
                StateField(
                    name="amplitude",
                    label="Amplitude",
                    type=Type.float,
                    default=0.5,
                    access=Access.rw,
                ),
                StateField(
                    name="phase",
                    label="Phase offset normalized (0-1)",
                    type=Type.float,
                    default=0.0,
                    access=Access.rw,
                ),
                StateField(
                    name="offset",
                    label="Vertical Offset",
                    type=Type.float,
                    default=0.5,
                    access=Access.rw,
                ),
            ],
            allowAddStates=False,
            commands=[],
            allowAddCommands=False,
        )
    )
    return registry


if __name__ == "__main__":
    import json
    from pathlib import Path
    registry = build_demo_registry()
    graph = OperatorGraph()

    try:
        graph_path = Path('graph.json')
        if graph_path.exists():
            graph_data = json.load(open(graph_path))
            graph.load_dict(graph_data['graph'])
    except Exception as e:
        print(f"Failed to load graph: {e}")
    
    OperatorGraphEditor(operatorCls_registry=registry, graph=graph).run()
