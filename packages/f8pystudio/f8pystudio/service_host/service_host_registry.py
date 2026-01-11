from f8pysdk import (
    F8ServiceSpec,
    F8ServiceSchemaVersion,
    F8OperatorSpec,
    F8OperatorSchemaVersion,
    F8DataPortSpec,
    any_schema,
    F8StateSpec,
    F8StateAccess,
    integer_schema,
)


class ServiceHostRegistry:
    """Registry for service host specifications and operator specifications."""

    def __init__(self):
        self._service_spec = F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass="f8.pystudio",
            version="0.0.1",
            label="Studio",
            description="Service Graph Editor in Python and Qt.",
            tags=["editor", "ui", "python", "py"],
            rendererClass="",
            states=[
                F8StateSpec(
                    name="tickMs",
                    label="Refresh Interval (ms)",
                    description="Interval in milliseconds for refreshing the UI nodes in the editor.",
                    valueSchema=integer_schema(default=100, minimum=16, maximum=5000),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
            ],
            editableStates=False,
            editableDataInPorts=False,
            editableDataOutPorts=False,
            editableCommands=False,
        )
        self._operator_spec_registry: dict[str, F8OperatorSpec] = {}

        # debug
        self._operator_spec_registry["ExampleOperator"] = F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass="f8.pystudio",
            operatorClass="f8.example_operator",
            version="0.0.1",
            label="Example Operator",
            description="An example operator for demonstration purposes.",
            tags=["example", "demo"],
        )

        self._operator_spec_registry["PrintNodeOperator"] = F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass="f8.pystudio",
            operatorClass="f8.print_node_operator",
            version="0.0.1",
            label="Example Operator",
            description="Operator that prints node information to the console.",
            tags=["print", "console"],
            dataInPorts=[
                F8DataPortSpec(
                    name="inputData",
                    description="Data input to be printed.",
                    valueSchema=any_schema(),
                ),
            ],
        )

    @staticmethod
    def instance() -> "ServiceHostRegistry":
        # Singleton instance accessor.
        if not hasattr(ServiceHostRegistry, "_instance"):
            ServiceHostRegistry._instance = ServiceHostRegistry()
        return ServiceHostRegistry._instance

    @property
    def serviceClass(self) -> str:
        return self._service_spec.serviceClass

    def service_spec(self) -> F8ServiceSpec:
        return self._service_spec

    def operator_specs(self) -> list[F8OperatorSpec]:
        return list(self._operator_spec_registry.values())

    def register_operator(self, spec: F8OperatorSpec):
        if self.serviceClass != spec.serviceClass:
            raise ValueError("Cannot register operator spec for different service class.")
        if spec.operatorClass in self._operator_spec_registry:
            raise ValueError(f"Operator spec for class '{spec.operatorClass}' is already registered.")
        self._operator_spec_registry[spec.operatorClass] = spec

    def get_operator(self, operator_class: str) -> F8OperatorSpec | None:
        return self._operator_spec_registry.get(operator_class)

    def __getitem__(self, operator_class: str) -> F8OperatorSpec:
        spec = self.get_operator(operator_class)
        if spec is None:
            raise KeyError(f"Operator spec for class '{operator_class}' not found.")
        return spec
