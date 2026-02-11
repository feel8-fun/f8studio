from __future__ import annotations

import json
import logging
from typing import Any

from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from .pystudio_node_registry import SERVICE_CLASS, register_pystudio_specs

logger = logging.getLogger(__name__)


class PyStudioProgram:
    def describe_json(self) -> dict[str, Any]:
        register_pystudio_specs()
        return RuntimeNodeRegistry.instance().describe(SERVICE_CLASS).model_dump(mode="json")

    @staticmethod
    def _ensure_pystudio_specs_in_catalog() -> None:
        try:
            reg = register_pystudio_specs()
            from .service_catalog import ServiceCatalog

            sc = ServiceCatalog.instance()
            svc = reg.service_spec(SERVICE_CLASS)
            if svc is not None:
                sc.register_service(svc)
            for op in reg.operator_specs(SERVICE_CLASS):
                sc.register_operator(op)
        except Exception:
            logger.exception("Failed to ensure pystudio specs in catalog")

    @staticmethod
    def build_node_classes() -> list[type]:
        from .render_nodes import RenderNodeRegistry
        from .service_catalog import ServiceCatalog
        from .nodegraph.missing_operator_basenode import F8StudioMissingOperatorBaseNode
        from .nodegraph.missing_service_basenode import F8StudioMissingServiceBaseNode
        from f8pysdk import (
            F8OperatorSchemaVersion,
            F8OperatorSpec,
            F8ServiceSchemaVersion,
            F8ServiceSpec,
            F8StateAccess,
            F8StateSpec,
        )
        from f8pysdk.schema_helpers import any_schema, string_schema

        render_node_reg = RenderNodeRegistry.instance()
        service_catalog = ServiceCatalog.instance()

        generated_node_cls: list[type] = []
        for svc in service_catalog.services.all():
            base_cls = render_node_reg.get(svc.rendererClass, fallback_key="default_svc")
            node_cls = type(
                svc.serviceClass,
                (base_cls,),
                {"__identifier__": "svc", "NODE_NAME": svc.label, "SPEC_TEMPLATE": svc},
            )
            generated_node_cls.append(node_cls)

        for op in service_catalog.operators.all():
            base_cls = render_node_reg.get(op.rendererClass, fallback_key="default_op")
            node_cls = type(
                op.operatorClass,
                (base_cls,),
                {"__identifier__": op.serviceClass, "NODE_NAME": op.label, "SPEC_TEMPLATE": op},
            )
            generated_node_cls.append(node_cls)

        # Always register a placeholder node class so sessions can load even if
        # discovery is missing some operator types.
        missing_spec = F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass=SERVICE_CLASS,
            operatorClass="f8.missing",
            version="0.0.1",
            label="Missing Operator",
            description="Placeholder for an operator node whose type is not registered in this Studio session.",
            tags=["__missing__"],
            dataInPorts=[],
            dataOutPorts=[],
            execInPorts=[],
            execOutPorts=[],
            editableDataInPorts=False,
            editableDataOutPorts=False,
            editableExecInPorts=False,
            editableExecOutPorts=False,
            editableStateFields=False,
            stateFields=[
                F8StateSpec(
                    name="missingType",
                    label="Missing Type",
                    description="Original `type_` string from the session file.",
                    valueSchema=string_schema(),
                    access=F8StateAccess.ro,
                    showOnNode=False,
                    required=False,
                ),
                F8StateSpec(
                    name="missingSpec",
                    label="Missing Spec",
                    description="Original `f8_spec` JSON from the session file.",
                    valueSchema=any_schema(),
                    access=F8StateAccess.ro,
                    showOnNode=False,
                    required=False,
                ),
            ],
        )
        missing_node_cls = type(
            "f8.missing",
            (F8StudioMissingOperatorBaseNode,),
            {"__identifier__": SERVICE_CLASS, "NODE_NAME": "Missing", "SPEC_TEMPLATE": missing_spec},
        )
        generated_node_cls.append(missing_node_cls)

        missing_service_spec = F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass="f8.missing",
            version="0.0.1",
            label="Missing Service",
            description="Placeholder for a service/container node whose type is not registered in this Studio session.",
            tags=["__missing__"],
            rendererClass="default_container",
            stateFields=[
                F8StateSpec(
                    name="missingType",
                    label="Missing Type",
                    description="Original `type_` string from the session file.",
                    valueSchema=string_schema(),
                    access=F8StateAccess.ro,
                    showOnNode=False,
                    required=False,
                ),
                F8StateSpec(
                    name="missingSpec",
                    label="Missing Spec",
                    description="Original `f8_spec` JSON from the session file.",
                    valueSchema=any_schema(),
                    access=F8StateAccess.ro,
                    showOnNode=False,
                    required=False,
                ),
            ],
            editableStateFields=False,
            editableDataInPorts=False,
            editableDataOutPorts=False,
            editableCommands=False,
        )
        missing_service_node_cls = type(
            "f8.missing",
            (F8StudioMissingServiceBaseNode,),
            {"__identifier__": "svc", "NODE_NAME": "Missing Service", "SPEC_TEMPLATE": missing_service_spec},
        )
        generated_node_cls.append(missing_service_node_cls)

        return generated_node_cls

    def run(self) -> int:
        # Local import: keep `--describe` fast and avoid importing Qt at module import time.
        from qtpy import QtWidgets

        from .widgets.main_window import F8StudioMainWin
        from .service_catalog import load_discovery_into_registries
        from .service_catalog.discovery import last_discovery_error_lines, last_discovery_timing_lines

        load_discovery_into_registries()
        self._ensure_pystudio_specs_in_catalog()

        node_classes = self.build_node_classes()

        app = QtWidgets.QApplication([])
        mainwin = F8StudioMainWin(node_classes)
        mainwin.show()

        try:
            timing_lines = last_discovery_timing_lines()
            for line in timing_lines:
                mainwin._bridge.log.emit(str(line))  # type: ignore[attr-defined]
            # Avoid double-printing errors: timings output can already include them.
            if not any("discovery errors:" in str(x) for x in timing_lines):
                for line in last_discovery_error_lines():
                    mainwin._bridge.log.emit(str(line))  # type: ignore[attr-defined]
        except Exception:
            logger.exception("Failed to emit discovery logs to studio log dock")
        return int(app.exec_() or 0)

    def describe_json_text(self) -> str:
        return json.dumps(self.describe_json(), ensure_ascii=False)
