from __future__ import annotations

from f8pysdk.specs import exec_port_specs

import time
from dataclasses import dataclass
from typing import Any, Final

from f8pysdk.codec import unwrap_json_value
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry
from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8SpecEditPolicy,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    editable_collection_edit_policy,
    string_schema,
)

from ..constants import SERVICE_CLASS
from ._ports import exec_out_ports

EXEC_BRANCH_OPERATOR_CLASS: Final[str] = "f8.exec_branch"
EXEC_MERGE_OPERATOR_CLASS: Final[str] = "f8.exec_merge"
DATA_MUX_OPERATOR_CLASS: Final[str] = "f8.data_mux"

_DEFAULT_BRANCH_PORTS: Final[list[str]] = ["branch_a", "branch_b", "branch_c", "default"]
_DEFAULT_MERGE_INPUTS: Final[list[str]] = ["branch_a", "branch_b", "branch_c"]
_DEFAULT_MUX_INPUTS: Final[list[str]] = ["branch_a", "branch_b", "branch_c", "default"]
_ERROR_LOG_INTERVAL_MS: Final[int] = 5000


def _port_names_from_data_ports(node: F8RuntimeNode, *, default: list[str]) -> list[str]:
    ports = [str(port.name) for port in list(node.dataInPorts or []) if str(port.name or "").strip()]
    if ports:
        return ports
    return list(default)


def _exec_in_port_names(node: F8RuntimeNode, *, default: list[str]) -> list[str]:
    ports = [str(port) for port in list(node.execInPorts or []) if str(port or "").strip()]
    if ports:
        return ports
    return list(default)


def _state_field_names(node: F8RuntimeNode, *, default: list[str]) -> list[str]:
    fields = [str(field.name) for field in list(node.stateFields or []) if str(field.name or "").strip()]
    if fields:
        return fields
    return list(default)


def _normalized_selector(value: Any) -> str:
    return str(unwrap_json_value(value) or "").strip()


@dataclass
class _DedupeReporter:
    last_fingerprint: str = ""
    last_ts_ms: int = 0

    def should_report(self, fingerprint: str, *, now_ms: int) -> bool:
        if fingerprint != self.last_fingerprint:
            self.last_fingerprint = str(fingerprint)
            self.last_ts_ms = int(now_ms)
            return True
        if int(now_ms) - int(self.last_ts_ms) >= _ERROR_LOG_INTERVAL_MS:
            self.last_ts_ms = int(now_ms)
            return True
        return False


class ExecBranchRuntimeNode(OperatorNode):
    """
    Mutually-exclusive exec branch.

    `selectedBranch` is a low-frequency semantic state value. Each exec trigger
    emits exactly one matching branch output, or `default` when configured.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[str(port.name) for port in list(node.dataInPorts or [])],
            data_out_ports=[str(port.name) for port in list(node.dataOutPorts or [])],
            state_fields=_state_field_names(node, default=["selectedBranch", "resolvedBranch"]),
            exec_in_ports=_exec_in_port_names(node, default=["exec"]),
            exec_out_ports=exec_out_ports(node, default=_DEFAULT_BRANCH_PORTS),
        )
        self._selected_branch = "branch_a"
        self._resolved_branch = ""
        self._last_published_resolved_branch = ""
        self._missing_branch_reporter = _DedupeReporter()
        self._refresh_state(dict(initial_state or {}))

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name != "selectedBranch":
            return
        self._selected_branch = _normalized_selector(value)

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "selectedBranch":
            return _normalized_selector(value)
        return value

    async def on_exec(self, _exec_id: str | int, _in_port: str | None = None) -> list[str]:
        resolved = self._resolve_branch()
        self._resolved_branch = resolved
        await self._publish_resolved_branch_if_needed()
        if not resolved:
            await self._report_missing_branch()
            return []
        return [resolved]

    def _refresh_state(self, values: dict[str, Any]) -> None:
        if "selectedBranch" in values:
            self._selected_branch = _normalized_selector(values.get("selectedBranch"))
        if "resolvedBranch" in values:
            self._resolved_branch = _normalized_selector(values.get("resolvedBranch"))
            self._last_published_resolved_branch = self._resolved_branch

    def _resolve_branch(self) -> str:
        selected = str(self._selected_branch or "").strip()
        if selected and selected in self.exec_out_ports:
            return selected
        if "default" in self.exec_out_ports:
            return "default"
        return ""

    async def _publish_resolved_branch_if_needed(self) -> None:
        if self._last_published_resolved_branch == self._resolved_branch:
            return
        self._last_published_resolved_branch = str(self._resolved_branch)
        await self.set_state("resolvedBranch", str(self._resolved_branch))

    async def _report_missing_branch(self) -> None:
        now_ms = int(time.time() * 1000.0)
        selected = str(self._selected_branch or "")
        fingerprint = f"exec_branch_missing:{self.node_id}:{selected}"
        if not self._missing_branch_reporter.should_report(fingerprint, now_ms=now_ms):
            return
        await self.report_error(
            "EXEC_BRANCH_NO_MATCH",
            f"selectedBranch has no matching exec output and no default output: {selected!r}",
            severity="warning",
            fingerprint=fingerprint,
            ts_ms=now_ms,
        )


ExecBranchRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.execution",
    operatorClass=EXEC_BRANCH_OPERATOR_CLASS,
    version="0.0.1",
    label="Exec Branch",
    description="Mutually-exclusive exec branch selected by low-frequency state.",
    tags=["execution", "branch", "switch", "mode"],
    execInPorts=exec_port_specs(["exec"]),
    execOutPorts=exec_port_specs(list(_DEFAULT_BRANCH_PORTS)),
    editPolicy=F8SpecEditPolicy(execOutPorts=editable_collection_edit_policy()),
    stateFields=[
        F8StateSpec(
            name="selectedBranch",
            label="Selected Branch",
            description="Exec output port to emit for each trigger.",
            valueSchema=string_schema(default="branch_a"),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="resolvedBranch",
            label="Resolved Branch",
            description="Readonly branch output actually emitted after fallback.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=True,
        ),
    ],
)


class ExecMergeRuntimeNode(OperatorNode):
    """
    Any-input exec merge.

    This is not a barrier. Any configured exec input emits the single `exec`
    output, preserving the core single-upstream-per-input-port rule.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        del initial_state
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[str(port.name) for port in list(node.dataInPorts or [])],
            data_out_ports=[str(port.name) for port in list(node.dataOutPorts or [])],
            state_fields=[str(field.name) for field in list(node.stateFields or [])],
            exec_in_ports=_exec_in_port_names(node, default=_DEFAULT_MERGE_INPUTS),
            exec_out_ports=exec_out_ports(node, default=["exec"]),
        )

    async def on_exec(self, _exec_id: str | int, _in_port: str | None = None) -> list[str]:
        if "exec" in self.exec_out_ports:
            return ["exec"]
        if self.exec_out_ports:
            return [str(self.exec_out_ports[0])]
        return []


ExecMergeRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.execution",
    operatorClass=EXEC_MERGE_OPERATOR_CLASS,
    version="0.0.1",
    label="Exec Merge",
    description="Merge mutually-exclusive exec branches into one continuation.",
    tags=["execution", "merge", "join", "branch"],
    execInPorts=exec_port_specs(list(_DEFAULT_MERGE_INPUTS)),
    execOutPorts=exec_port_specs(["exec"]),
    editPolicy=F8SpecEditPolicy(execInPorts=editable_collection_edit_policy()),
)


class DataMuxRuntimeNode(OperatorNode):
    """
    Pull-based data mux.

    `selectedInput` should usually be driven by the same low-frequency mode state
    that drives Exec Branch.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=_port_names_from_data_ports(node, default=_DEFAULT_MUX_INPUTS),
            data_out_ports=[str(port.name) for port in list(node.dataOutPorts or [])] or ["out"],
            state_fields=_state_field_names(node, default=["selectedInput", "resolvedInput"]),
            exec_in_ports=_exec_in_port_names(node, default=["exec"]),
            exec_out_ports=exec_out_ports(node, default=["exec"]),
        )
        self._selected_input = "branch_a"
        self._resolved_input = ""
        self._last_published_resolved_input = ""
        self._last_ctx_id: str | int | None = None
        self._cache_valid = False
        self._cached_outputs: dict[str, Any] = {}
        self._missing_input_reporter = _DedupeReporter()
        self._refresh_state(dict(initial_state or {}))

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name != "selectedInput":
            return
        self._selected_input = _normalized_selector(value)
        self._cache_valid = False

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "selectedInput":
            return _normalized_selector(value)
        return value

    async def on_exec(self, _exec_id: str | int, _in_port: str | None = None) -> list[str]:
        if "exec" in self.exec_out_ports:
            return ["exec"]
        if self.exec_out_ports:
            return [str(self.exec_out_ports[0])]
        return []

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        out_port = str(port or "")
        if out_port not in self.data_out_ports:
            return None
        if self._cache_valid and ctx_id is not None and ctx_id == self._last_ctx_id:
            return self._cached_outputs.get(out_port)

        value = await self._compute_selected_value(ctx_id=ctx_id)
        if value is None:
            self._cached_outputs = {}
        else:
            self._cached_outputs = {self._default_output_port(): value}
        self._last_ctx_id = ctx_id
        self._cache_valid = ctx_id is not None
        return self._cached_outputs.get(out_port)

    def _refresh_state(self, values: dict[str, Any]) -> None:
        if "selectedInput" in values:
            self._selected_input = _normalized_selector(values.get("selectedInput"))
        if "resolvedInput" in values:
            self._resolved_input = _normalized_selector(values.get("resolvedInput"))
            self._last_published_resolved_input = self._resolved_input

    def _default_output_port(self) -> str:
        if "out" in self.data_out_ports:
            return "out"
        if self.data_out_ports:
            return str(self.data_out_ports[0])
        return "out"

    def _resolve_input_port(self) -> str:
        selected = str(self._selected_input or "").strip()
        if selected and selected in self.data_in_ports:
            return selected
        if "default" in self.data_in_ports:
            return "default"
        return ""

    async def _compute_selected_value(self, *, ctx_id: str | int | None) -> Any:
        resolved = self._resolve_input_port()
        self._resolved_input = resolved
        await self._publish_resolved_input_if_needed()
        if not resolved:
            await self._report_missing_input()
            return None
        value = await self.pull(resolved, ctx_id=ctx_id)
        if value is None and resolved != "default" and "default" in self.data_in_ports:
            self._resolved_input = "default"
            await self._publish_resolved_input_if_needed()
            return await self.pull("default", ctx_id=ctx_id)
        if value is None:
            await self._report_missing_input()
        return value

    async def _publish_resolved_input_if_needed(self) -> None:
        if self._last_published_resolved_input == self._resolved_input:
            return
        self._last_published_resolved_input = str(self._resolved_input)
        await self.set_state("resolvedInput", str(self._resolved_input))

    async def _report_missing_input(self) -> None:
        now_ms = int(time.time() * 1000.0)
        selected = str(self._selected_input or "")
        fingerprint = f"data_mux_missing:{self.node_id}:{selected}:{self._resolved_input}"
        if not self._missing_input_reporter.should_report(fingerprint, now_ms=now_ms):
            return
        await self.report_error(
            "DATA_MUX_NO_VALUE",
            f"selectedInput has no available value: selected={selected!r} resolved={self._resolved_input!r}",
            severity="warning",
            fingerprint=fingerprint,
            ts_ms=now_ms,
        )


DataMuxRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.flow",
    operatorClass=DATA_MUX_OPERATOR_CLASS,
    version="0.0.1",
    label="Data Mux",
    description="Select one data input by low-frequency state and expose it as one output.",
    tags=["data", "mux", "switch", "branch", "mode"],
    execInPorts=exec_port_specs(["exec"]),
    execOutPorts=exec_port_specs(["exec"]),
    dataInPorts=[
        F8DataPortSpec(name="branch_a", description="Branch A input.", valueSchema=any_schema(), definitionProtected=False),
        F8DataPortSpec(name="branch_b", description="Branch B input.", valueSchema=any_schema(), definitionProtected=False),
        F8DataPortSpec(name="branch_c", description="Branch C input.", valueSchema=any_schema(), definitionProtected=False),
        F8DataPortSpec(name="default", description="Fallback input.", valueSchema=any_schema(), definitionProtected=False),
    ],
    dataOutPorts=[
        F8DataPortSpec(name="out", description="Selected data output.", valueSchema=any_schema(), definitionProtected=False),
    ],
    editPolicy=F8SpecEditPolicy(dataInPorts=editable_collection_edit_policy()),
    stateFields=[
        F8StateSpec(
            name="selectedInput",
            label="Selected Input",
            description="Data input port to pull for the selected output.",
            valueSchema=string_schema(default="branch_a"),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="resolvedInput",
            label="Resolved Input",
            description="Readonly input port actually pulled after fallback.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=True,
        ),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(ExecBranchRuntimeNode.SPEC, ExecBranchRuntimeNode, overwrite=True)
    registry.register_operator(ExecMergeRuntimeNode.SPEC, ExecMergeRuntimeNode, overwrite=True)
    registry.register_operator(DataMuxRuntimeNode.SPEC, DataMuxRuntimeNode, overwrite=True)
    return registry


__all__ = [
    "DataMuxRuntimeNode",
    "ExecBranchRuntimeNode",
    "ExecMergeRuntimeNode",
    "register_operator",
]
