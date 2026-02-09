from __future__ import annotations

import json
from typing import Any

from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from .pystudio_node_registry import SERVICE_CLASS, register_pystudio_specs


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
            pass

    @staticmethod
    def build_node_classes() -> list[type]:
        from .render_nodes import RenderNodeRegistry
        from .service_catalog import ServiceCatalog

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
                try:
                    mainwin._bridge.log.emit(str(line))  # type: ignore[attr-defined]
                except Exception:
                    pass
            # Avoid double-printing errors: timings output can already include them.
            if not any("discovery errors:" in str(x) for x in timing_lines):
                for line in last_discovery_error_lines():
                    try:
                        mainwin._bridge.log.emit(str(line))  # type: ignore[attr-defined]
                    except Exception:
                        pass
        except Exception:
            pass
        return int(app.exec_() or 0)

    def describe_json_text(self) -> str:
        return json.dumps(self.describe_json(), ensure_ascii=False)
