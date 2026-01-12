from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from NodeGraphQt import BaseNode
from NodeGraphQt.constants import NodePropWidgetEnum

from f8pysdk import F8PrimitiveTypeEnum, F8ServiceSpec, F8StateAccess, schema_default, schema_type

from ..schema.compat import PORT_KIND_DATA_KEY, PORT_SCHEMA_SIG_DATA_KEY, schema_signature
from ..services.service_registry import ServiceSpecRegistry

from .generic import DATA_PORT_COLOR, STATE_PORT_COLOR, GridNodeItem, PORT_KIND_DATA, PORT_KIND_STATE, PORT_ROW_DATA_KEY


@dataclass
class ServicePortHandles:
    data_in: dict[str, Any] = field(default_factory=dict)
    data_out: dict[str, Any] = field(default_factory=dict)
    state_in: dict[str, Any] = field(default_factory=dict)
    state_out: dict[str, Any] = field(default_factory=dict)


class ServiceNode(BaseNode):  # type: ignore[misc]
    """
    Default service node renderer (v1).

    - Looks like an operator node (not a backdrop).
    - Has data/state ports (no exec ports).
    - Spec source: ServiceSpecRegistry via `SPEC_KEY == serviceClass`.
    """

    __identifier__ = "feel8.service"
    NODE_NAME = "Service"

    SPEC_KEY: str = ""
    spec: F8ServiceSpec

    def __init__(self) -> None:
        super().__init__(qgraphics_item=GridNodeItem)
        stable_id = uuid.uuid4().hex
        try:
            self.model.id = stable_id  # type: ignore[attr-defined]
            self.view.id = stable_id  # type: ignore[attr-defined]
        except Exception:
            pass

        spec_key = str(self.SPEC_KEY or "").strip()
        if not spec_key:
            raise ValueError("ServiceNode.SPEC_KEY is empty")
        self.spec = ServiceSpecRegistry.instance().get(spec_key)

        self.port_handles = ServicePortHandles()

        # Required for dynamic port rebuilds (`delete_input`/`delete_output`).
        try:
            self.set_port_deletion_allowed(True)  # type: ignore[attr-defined]
        except Exception:
            pass

        self._build_ports()
        self._apply_state_properties()

        try:
            self.set_name(self.spec.label)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _tag_port_row(self, handle: Any, row: int) -> None:
        try:
            handle.view.setData(PORT_ROW_DATA_KEY, int(row))
        except Exception:
            pass

    def _tag_port_meta(self, handle: Any, *, kind: str, schema: Any | None = None) -> None:
        try:
            handle.view.setData(PORT_KIND_DATA_KEY, str(kind))
        except Exception:
            pass
        try:
            handle.view.setData(PORT_SCHEMA_SIG_DATA_KEY, schema_signature(schema))
        except Exception:
            pass

    def _build_ports(self) -> None:
        # clear existing ports
        try:
            for p in list(self.input_ports()):  # type: ignore[attr-defined]
                self.delete_input(p)
            for p in list(self.output_ports()):  # type: ignore[attr-defined]
                self.delete_output(p)
        except Exception:
            pass

        self.port_handles = ServicePortHandles()

        row_offset = 0

        data_in = list(self.spec.dataInPorts or [])
        data_out = list(self.spec.dataOutPorts or [])
        rows = max(len(data_in), len(data_out))
        for idx, port in enumerate(data_in):
            handle = self.add_input(f"[D]{port.name}", color=DATA_PORT_COLOR)  # type: ignore[attr-defined]
            self._tag_port_row(handle, row_offset + idx)
            self._tag_port_meta(handle, kind=PORT_KIND_DATA, schema=port.valueSchema)
            self.port_handles.data_in[port.name] = handle
        for idx, port in enumerate(data_out):
            handle = self.add_output(f"{port.name}[D]", color=DATA_PORT_COLOR)  # type: ignore[attr-defined]
            self._tag_port_row(handle, row_offset + idx)
            self._tag_port_meta(handle, kind=PORT_KIND_DATA, schema=port.valueSchema)
            self.port_handles.data_out[port.name] = handle
        row_offset += rows

        fields = list(self.spec.stateFields or [])
        for idx, field_def in enumerate(fields):
            access = field_def.access or F8StateAccess.ro
            row = row_offset + idx

            if access != F8StateAccess.ro:
                handle = self.add_input(  # type: ignore[attr-defined]
                    name=f"[S]{field_def.name}",
                    color=STATE_PORT_COLOR,
                    display_name=False,
                )
                self._tag_port_row(handle, row)
                self._tag_port_meta(handle, kind=PORT_KIND_STATE, schema=field_def.valueSchema)
                self.port_handles.state_in[field_def.name] = handle

            if access != F8StateAccess.wo:
                handle = self.add_output(  # type: ignore[attr-defined]
                    name=f"{field_def.name}[S]",
                    color=STATE_PORT_COLOR,
                    display_name=False,
                )
                self._tag_port_row(handle, row)
                self._tag_port_meta(handle, kind=PORT_KIND_STATE, schema=field_def.valueSchema)
                self.port_handles.state_out[field_def.name] = handle

        try:
            self.view.draw_node()
        except Exception:
            pass

    @contextmanager
    def _atomic_node_update(self):
        viewer = None
        viewport = None
        try:
            graph = getattr(self, "graph", None)
            if graph is not None:
                try:
                    viewer = graph.viewer()
                    viewport = viewer.viewport() if viewer is not None else None
                except Exception:
                    viewer = None
                    viewport = None

            if viewer is not None:
                try:
                    viewer.setUpdatesEnabled(False)
                except Exception:
                    pass
            if viewport is not None:
                try:
                    viewport.setUpdatesEnabled(False)
                except Exception:
                    pass

            try:
                self.view.begin_transaction()  # type: ignore[attr-defined]
            except Exception:
                pass

            yield
        finally:
            try:
                self.view.end_transaction()  # type: ignore[attr-defined]
            except Exception:
                try:
                    self.view.draw_node()
                except Exception:
                    pass

            if viewport is not None:
                try:
                    viewport.setUpdatesEnabled(True)
                except Exception:
                    pass
            if viewer is not None:
                try:
                    viewer.setUpdatesEnabled(True)
                    viewer.update()
                except Exception:
                    pass

    def _apply_state_properties(self) -> None:
        for field_def in self.spec.stateFields or []:
            default_value = schema_default(field_def.valueSchema)
            field_type = schema_type(field_def.valueSchema)
            if self.has_property(field_def.name):  # type: ignore[attr-defined]
                continue

            if field_type == F8PrimitiveTypeEnum.boolean:
                self.create_property(field_def.name, default_value, widget_type=NodePropWidgetEnum.QCHECK_BOX.value)
            elif field_type == F8PrimitiveTypeEnum.integer:
                self.create_property(field_def.name, default_value, widget_type=NodePropWidgetEnum.QLINE_EDIT.value)
            elif field_type == F8PrimitiveTypeEnum.number:
                self.create_property(field_def.name, default_value, widget_type=NodePropWidgetEnum.QLINE_EDIT.value)
            elif field_type == F8PrimitiveTypeEnum.string:
                enum_values = getattr(field_def.valueSchema, "enum", None) if field_def.valueSchema else None
                if enum_values:
                    items = [str(v) for v in enum_values]
                    self.create_property(field_def.name, default_value, widget_type=NodePropWidgetEnum.QCOMBO_BOX.value, items=items)
                else:
                    self.create_property(field_def.name, default_value, widget_type=NodePropWidgetEnum.QLINE_EDIT.value)
            else:
                self.create_property(field_def.name, default_value, widget_type=NodePropWidgetEnum.QLINE_EDIT.value)

    def ensure_state_properties(self) -> None:
        self._apply_state_properties()

    def apply_spec(self, spec: F8ServiceSpec) -> None:
        self._validate_spec_for_ports(spec)
        old_spec = getattr(self, "spec", None)
        edge_snapshots = self._snapshot_edges()
        with self._atomic_node_update():
            try:
                self.spec = spec
                try:
                    self.set_name(self.spec.label)  # type: ignore[attr-defined]
                except Exception:
                    pass
                self._build_ports()
                self._apply_state_properties()
                self._restore_edges(edge_snapshots)
            except Exception:
                if old_spec is not None:
                    try:
                        self.spec = old_spec
                        try:
                            self.set_name(self.spec.label)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        self._build_ports()
                        self._apply_state_properties()
                        self._restore_edges(edge_snapshots)
                    except Exception:
                        pass
                raise

    def _spec_port_signature(self, spec: F8ServiceSpec) -> dict[str, Any]:
        sig: dict[str, Any] = {}
        for port in spec.dataInPorts or []:
            try:
                sig[f"[D]{port.name}"] = schema_type(port.valueSchema)
            except Exception:
                sig[f"[D]{port.name}"] = None
        for port in spec.dataOutPorts or []:
            try:
                sig[f"{port.name}[D]"] = schema_type(port.valueSchema)
            except Exception:
                sig[f"{port.name}[D]"] = None

        for field in spec.stateFields or []:
            access = field.access or F8StateAccess.ro
            try:
                st = schema_type(field.valueSchema)
            except Exception:
                st = None
            if access != F8StateAccess.ro:
                sig[f"[S]{field.name}"] = st
            if access != F8StateAccess.wo:
                sig[f"{field.name}[S]"] = st
        return sig

    def _port_signature_for_node(self, node: Any, port_name: str) -> Any:
        try:
            spec = getattr(node, "spec", None)
            if spec is None:
                return None
            if hasattr(node, "_spec_port_signature"):
                mapping = node._spec_port_signature(spec)  # type: ignore[attr-defined]
                return mapping.get(port_name)
        except Exception:
            return None
        return None

    def _snapshot_edges(self) -> set[tuple[str, str, str, str, Any, Any]]:
        edges: set[tuple[str, str, str, str]] = set()
        for port in [*list(self.inputs().values()), *list(self.outputs().values())]:  # type: ignore[attr-defined]
            try:
                connected = port.connected_ports()
            except Exception:
                connected = []
            for other in connected:
                try:
                    a_is_out = port.type_() == "out"
                    b_is_out = other.type_() == "out"
                    if a_is_out and not b_is_out:
                        edges.add((port.node().id, port.name(), other.node().id, other.name()))
                    elif b_is_out and not a_is_out:
                        edges.add((other.node().id, other.name(), port.node().id, port.name()))
                except Exception:
                    continue

        snapshots: set[tuple[str, str, str, str, Any, Any]] = set()
        try:
            graph = self.graph
        except Exception:
            graph = None
        for src_id, src_port, dst_id, dst_port in edges:
            if graph is None:
                snapshots.add((src_id, src_port, dst_id, dst_port, None, None))
                continue
            try:
                src_node = graph.get_node_by_id(src_id)
                dst_node = graph.get_node_by_id(dst_id)
            except Exception:
                src_node = None
                dst_node = None
            src_sig = self._port_signature_for_node(src_node, src_port) if src_node is not None else None
            dst_sig = self._port_signature_for_node(dst_node, dst_port) if dst_node is not None else None
            snapshots.add((src_id, src_port, dst_id, dst_port, src_sig, dst_sig))
        return snapshots

    def _restore_edges(self, edge_snapshots: set[tuple[str, str, str, str, Any, Any]]) -> None:
        try:
            graph = self.graph
        except Exception:
            return

        for src_id, src_port_name, dst_id, dst_port_name, src_sig, dst_sig in edge_snapshots:
            try:
                src_node = graph.get_node_by_id(src_id)
                dst_node = graph.get_node_by_id(dst_id)
            except Exception:
                continue
            if src_node is None or dst_node is None:
                continue

            if src_sig is not None:
                current_src_sig = self._port_signature_for_node(src_node, src_port_name)
                if current_src_sig != src_sig:
                    continue
            if dst_sig is not None:
                current_dst_sig = self._port_signature_for_node(dst_node, dst_port_name)
                if current_dst_sig != dst_sig:
                    continue

            try:
                out_port = src_node.outputs().get(src_port_name)
                in_port = dst_node.inputs().get(dst_port_name)
            except Exception:
                continue
            if out_port is None or in_port is None:
                continue

            try:
                out_port.connect_to(in_port)
            except Exception:
                pass

    @staticmethod
    def _find_duplicates(values: list[str]) -> list[str]:
        seen: set[str] = set()
        dupes: set[str] = set()
        for value in values:
            if value in seen:
                dupes.add(value)
            else:
                seen.add(value)
        return sorted(dupes)

    def _validate_spec_for_ports(self, spec: F8ServiceSpec) -> None:
        data_in = [str(p.name).strip() for p in (spec.dataInPorts or [])]
        data_out = [str(p.name).strip() for p in (spec.dataOutPorts or [])]
        states = [str(s.name).strip() for s in (spec.stateFields or [])]

        errors: list[str] = []
        if any(not p for p in data_in):
            errors.append("dataInPorts contains an empty name.")
        if any(not p for p in data_out):
            errors.append("dataOutPorts contains an empty name.")
        if any(not p for p in states):
            errors.append("states contains an empty name.")

        dup = self._find_duplicates(data_in)
        if dup:
            errors.append(f"Duplicate dataInPorts: {', '.join(dup)}")
        dup = self._find_duplicates(data_out)
        if dup:
            errors.append(f"Duplicate dataOutPorts: {', '.join(dup)}")
        dup = self._find_duplicates(states)
        if dup:
            errors.append(f"Duplicate states: {', '.join(dup)}")

        if errors:
            raise ValueError("\n".join(errors))
