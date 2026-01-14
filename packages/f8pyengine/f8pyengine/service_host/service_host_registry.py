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
    number_schema,
    string_schema,
)


class ServiceHostRegistry:
    """Registry for service host specifications and operator specifications."""

    def __init__(self):
        self._service_spec = F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass="f8.pyengine",
            version="0.0.1",
            label="PyEngine",
            description="Python-based execution engine for Feel8 operators.",
            tags=["engine", "python", "py"],
            rendererClass="default_container",
            editableStateFields=False,
            editableDataInPorts=False,
            editableDataOutPorts=False,
            editableCommands=False,
        )
        self._operator_spec_registry: dict[str, F8OperatorSpec] = {}

        # Built-in demo operators
        self._operator_spec_registry["Tick"] = F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass="f8.pyengine",
            operatorClass="f8.tick",
            version="0.0.1",
            label="Tick",
            description="Tick operator that generates periodic ticks.",
            tags=["execution", "timer", "start", "clock", "entrypoint"],
            stateFields=[
                F8StateSpec(
                    name="tickMs",
                    label="Refresh Interval (ms)",
                    description="Interval in milliseconds for refreshing the execution engine cycle.",
                    valueSchema=integer_schema(default=100, minimum=1, maximum=50000),
                    access=F8StateAccess.wo,
                    showOnNode=True,
                ),
            ],
            execOutPorts=["exec"],
        )

        self._operator_spec_registry["Sine"] = F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass="f8.pyengine",
            operatorClass="f8.sine",
            version="0.0.1",
            label="Sine",
            description="Exec-driven sine generator (emits `value`).",
            tags=["signal", "sin", "waveform", "generator", "oscillator"],
            execInPorts=["exec"],
            dataOutPorts=[F8DataPortSpec(name="value", description="sine output", valueSchema=number_schema())],
            stateFields=[
                F8StateSpec(
                    name="hz",
                    label="Hz",
                    description="Frequency in Hz.",
                    valueSchema=number_schema(default=1.0, minimum=0.0, maximum=100.0),
                    access=F8StateAccess.wo,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="amp",
                    label="Amplitude",
                    description="Amplitude.",
                    valueSchema=number_schema(default=1.0, minimum=0.0, maximum=1000.0),
                    access=F8StateAccess.wo,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="phase",
                    label="Phase",
                    description="Normalized phase offset (0.0 - 1.0).",
                    valueSchema=number_schema(default=0.0, minimum=0.0, maximum=1.0),
                    access=F8StateAccess.wo,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="offset",
                    label="Offset",
                    description="DC offset.",
                    valueSchema=number_schema(default=0.0, minimum=-1000.0, maximum=1000.0),
                    access=F8StateAccess.wo,
                    showOnNode=True,
                ),
            ],
        )

        self._operator_spec_registry["Print"] = F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass="f8.pyengine",
            operatorClass="f8.print",
            version="0.0.1",
            label="Print",
            description="Prints incoming `value` samples.",
            tags=["debug", "console", "print"],
            dataInPorts=[F8DataPortSpec(name="value", description="value to print", valueSchema=number_schema())],
            stateFields=[
                F8StateSpec(
                    name="throttleMs",
                    label="Throttle (ms)",
                    description="Throttle interval in milliseconds.",
                    valueSchema=integer_schema(default=100, minimum=0, maximum=10000),
                    access=F8StateAccess.wo,
                    showOnNode=True,
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
