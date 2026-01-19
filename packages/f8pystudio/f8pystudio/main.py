import argparse
import asyncio
import json

from f8pysdk.runtime_node_registry import RuntimeNodeRegistry
from .service_host.service_host_registry import SERVICE_CLASS, ServiceHostRegistry
from .service_catalog import load_discovery_into_registries, ServiceCatalog


from qtpy import QtWidgets, QtCore

from .render_nodes import RenderNodeRegistry

# from .deploy import deploy_to_service, export_runtime_graph
from .service_process_manager import ServiceProcessManager, ServiceProcessConfig
from .widgets.main_window import F8StudioMainWin


def generate_node_classes():
    """Generate node classes from service catalog."""
    renderNodeReg = RenderNodeRegistry.instance()
    serviceCatalog = ServiceCatalog.instance()

    generated_node_cls = []

    for svc in serviceCatalog.services.all():
        print(f'Registering render nodes for service "{svc.serviceClass}"')

        base_cls = renderNodeReg.get(
            svc.rendererClass, fallback_key="default_svc"
        )  # Ensure service renderer is registered
        node_cls = type(
            svc.serviceClass,
            (base_cls,),
            {"__identifier__": "svc", "NODE_NAME": svc.label, "SPEC_TEMPLATE": svc},
        )
        generated_node_cls.append(node_cls)

    for op in serviceCatalog.operators.all():
        print(f'Registering render node for operator "{op.operatorClass}" in service "{op.serviceClass}"')
        # For demo purposes, we just register the GenericRenderNode for all operators.

        base_cls = renderNodeReg.get(op.rendererClass, fallback_key="default_op")
        node_cls = type(
            op.operatorClass,
            (base_cls,),
            {"__identifier__": op.serviceClass, "NODE_NAME": op.label, "SPEC_TEMPLATE": op},
        )
        generated_node_cls.append(node_cls)

    return generated_node_cls

def _main() -> int:
    """F8PyStudio main entry point."""
    parser = argparse.ArgumentParser(description="F8PyStudio Main")
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Output the service description in JSON format",
    )
    args = parser.parse_args()

    if args.describe:
        # Ensure f8.pystudio specs are registered into the shared registry.
        ServiceHostRegistry.instance()
        describe = RuntimeNodeRegistry.instance().describe(SERVICE_CLASS).model_dump(mode="json")

        print(json.dumps(describe, ensure_ascii=False))
        raise SystemExit(0)

    ret = load_discovery_into_registries()
    # Ensure pystudio internal nodes are available in the editor palette.
    try:
        sh = ServiceHostRegistry.instance()
        sc = ServiceCatalog.instance()
        sc.register_service(sh.service_spec())
        for op in sh.operator_specs():
            sc.register_operator(op)
    except Exception:
        pass

    # TODO: Generate renderer classes
    generated_node_cls = generate_node_classes()

    # Simple NodeGraphQt demo
    app = QtWidgets.QApplication([])

    mainwin = F8StudioMainWin(generated_node_cls)
    mainwin.show()

    # def _deploy() -> None:
    #     try:
    #         service_id, ok = QtWidgets.QInputDialog.getText(mainwin, "Deploy", "Target serviceId")
    #         if not ok:
    #             return
    #         service_id = str(service_id).strip()
    #         if not service_id:
    #             return
    #         nats_url, ok = QtWidgets.QInputDialog.getText(mainwin, "Deploy", "NATS URL", text="nats://127.0.0.1:4222")
    #         if not ok:
    #             return
    #         nats_url = str(nats_url).strip()

    #         rt = export_runtime_graph(studio_graph, service_id=service_id)
    #         asyncio.run(deploy_to_service(service_id=service_id, nats_url=nats_url, graph=rt))
    #         QtWidgets.QMessageBox.information(mainwin, "Deploy", f"Deployed to {service_id}")
    #     except Exception as exc:
    #         QtWidgets.QMessageBox.critical(mainwin, "Deploy failed", str(exc))

    # deploy_action = QtWidgets.QAction("Deploy…", mainwin)
    # deploy_action.triggered.connect(_deploy)  # type: ignore[attr-defined]

    proc_mgr = ServiceProcessManager()

    # def _selected_service_container() -> Any | None:
    #     try:
    #         sel = list(studio_graph.selected_nodes() or [])
    #     except Exception:
    #         sel = []
    #     for n in sel:
    #         spec = getattr(n, "spec", None)
    #         if spec is None:
    #             continue
    #         if str(getattr(spec, "serviceClass", "") or "") == "f8.pyengine":
    #             return n
    #     return None

    # def _service_id_for_container(n: Any) -> str:
    #     try:
    #         sid = str(getattr(n, "id", "") or "")
    #     except Exception:
    #         sid = ""
    #     sid = sid.replace(".", "_").strip()
    #     if not sid:
    #         sid = "engine1"
    #     return sid

    # def _run_service() -> None:
    #     n = _selected_service_container()
    #     if n is None:
    #         return
    #     service_id = _service_id_for_container(n)
    #     try:
    #         proc_mgr.start(ServiceProcessConfig(service_class="f8.pyengine", service_id=service_id))
    #     except Exception as exc:
    #         QtWidgets.QMessageBox.critical(mainwin, "Run failed", str(exc))

    # def _stop_service() -> None:
    #     n = _selected_service_container()
    #     if n is None:
    #         return
    #     proc_mgr.stop(_service_id_for_container(n))

    # def _deploy_selected() -> None:
    #     n = _selected_service_container()
    #     if n is None:
    #         return
    #     service_id = _service_id_for_container(n)
    #     try:
    #         nats_url = "nats://127.0.0.1:4222"
    #         nodes = []
    #         try:
    #             nodes = list(getattr(n, "contained_nodes")() or [])
    #         except Exception:
    #             nodes = []
    #         rt = export_runtime_graph(studio_graph, service_id=service_id, include_nodes=nodes)
    #         asyncio.run(deploy_to_service(service_id=service_id, nats_url=nats_url, graph=rt))
    #         QtWidgets.QMessageBox.information(mainwin, "Deploy", f"Deployed to {service_id}")
    #     except Exception as exc:
    #         QtWidgets.QMessageBox.critical(mainwin, "Deploy failed", str(exc))

    # def _deploy_and_run() -> None:
    #     _run_service()
    #     _deploy_selected()

    # Future: add Service/Deploy actions here (main window already owns Graph menu).

    app.exec_()


if __name__ == "__main__":
    _main()
