import argparse
import asyncio
import json
from f8pysdk import F8ServiceDescribe
from f8pystudio.service_host import ServiceHostRegistry
from f8pystudio.service_catalog import load_discovery_into_registries, ServiceCatalog


from qtpy import QtWidgets, QtCore
from NodeGraphQt import (
    NodeGraph,
    NodesPaletteWidget,
    NodesTreeWidget,
    BaseNode,
    BackdropNode,
    GroupNode,
)


from f8pystudio.widgets.palette_widget import F8NodesPaletteWidget

from f8pystudio.renderNodes import RenderNodeRegistry
from f8pystudio.deploy import deploy_to_service, export_runtime_graph
from f8pystudio.widgets.node_property_editor import NodePropertyEditorWidget
from f8pystudio.service_process_manager import ServiceProcessManager, ServiceProcessConfig


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
        describe = F8ServiceDescribe(
            service=ServiceHostRegistry.instance().service_spec(),
            operators=ServiceHostRegistry.instance().operator_specs(),
        ).model_dump(mode="json")

        print(json.dumps(describe, ensure_ascii=False))
        raise SystemExit(0)

    ret = load_discovery_into_registries()


    # TODO: Generate renderer classes

    renderNodeReg = RenderNodeRegistry.instance()
    serviceCatalog = ServiceCatalog.instance()
    
    generated_node_cls = []

    for svc in serviceCatalog.services.all():
        print(f'Registering render nodes for service "{svc.serviceClass}"')

        base_cls = renderNodeReg.get(svc.rendererClass)  # Ensure service renderer is registered
        node_cls = type(
            svc.serviceClass,
            (base_cls,),
            {"__identifier__": "svc", "NODE_NAME": svc.label, "spec": svc},
        )
        generated_node_cls.append(node_cls)


    for op in serviceCatalog.operators.all():
        print(f'Registering render node for operator "{op.operatorClass}" in service "{op.serviceClass}"')
        # For demo purposes, we just register the GenericRenderNode for all operators.

        base_cls = renderNodeReg.get(op.rendererClass)
        node_cls = type(
            op.operatorClass,
            (base_cls,),
            {"__identifier__": op.serviceClass, "NODE_NAME": op.label, "spec": op},
        )
        generated_node_cls.append(node_cls)


    print(
        f"Loaded {len(serviceCatalog.services.all())} services and {len(serviceCatalog.operators.query(None))} operators."
    )

    # Simple NodeGraphQt demo
    app = QtWidgets.QApplication([])

    mainwin = QtWidgets.QMainWindow()
    mainwin.setWindowTitle("F8PyStudio - NodeGraphQt Demo")
    mainwin.resize(1200, 800)
    mainwin.show()

    graph = NodeGraph()
    graph.node_factory.clear_registered_nodes()

    for cls in generated_node_cls:
        graph.node_factory.register_node(cls)

    palette = F8NodesPaletteWidget(node_graph=graph)
    # tree = NodesTreeWidget(node_graph=graph)

    mainwin.setCentralWidget(graph.widget)

    prop_editor = NodePropertyEditorWidget(node_graph=graph)
    prop_dock = QtWidgets.QDockWidget("Properties", mainwin)
    prop_dock.setWidget(prop_editor)
    mainwin.addDockWidget(QtCore.Qt.LeftDockWidgetArea, prop_dock)

    palette._category_tabs

    dock = QtWidgets.QDockWidget("Nodes Palette", mainwin)
    dock.setWidget(palette)
    mainwin.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    # dock2 = QtWidgets.QDockWidget("Nodes Tree", mainwin)
    # dock2.setWidget(tree)
    # mainwin.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock2)

    def _deploy() -> None:
        try:
            service_id, ok = QtWidgets.QInputDialog.getText(mainwin, "Deploy", "Target serviceId")
            if not ok:
                return
            service_id = str(service_id).strip()
            if not service_id:
                return
            nats_url, ok = QtWidgets.QInputDialog.getText(mainwin, "Deploy", "NATS URL", text="nats://127.0.0.1:4222")
            if not ok:
                return
            nats_url = str(nats_url).strip()

            rt = export_runtime_graph(graph, service_id=service_id)
            asyncio.run(deploy_to_service(service_id=service_id, nats_url=nats_url, graph=rt))
            QtWidgets.QMessageBox.information(mainwin, "Deploy", f"Deployed to {service_id}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(mainwin, "Deploy failed", str(exc))

    deploy_action = QtWidgets.QAction("Deploy…", mainwin)
    deploy_action.triggered.connect(_deploy)  # type: ignore[attr-defined]

    proc_mgr = ServiceProcessManager()

    def _selected_service_container() -> Any | None:
        try:
            sel = list(graph.selected_nodes() or [])
        except Exception:
            sel = []
        for n in sel:
            spec = getattr(n, "spec", None)
            if spec is None:
                continue
            if str(getattr(spec, "serviceClass", "") or "") == "f8.pyengine":
                return n
        return None

    def _service_id_for_container(n: Any) -> str:
        try:
            sid = str(getattr(n, "id", "") or "")
        except Exception:
            sid = ""
        sid = sid.replace(".", "_").strip()
        if not sid:
            sid = "engine1"
        return sid

    def _wrap_selected_into_container() -> None:
        n = _selected_service_container()
        if n is None:
            return
        try:
            wrap = getattr(n, "wrap_selected_nodes", None)
            if callable(wrap):
                wrap()
        except Exception:
            return

    def _run_service() -> None:
        n = _selected_service_container()
        if n is None:
            return
        service_id = _service_id_for_container(n)
        try:
            proc_mgr.start(ServiceProcessConfig(service_class="f8.pyengine", service_id=service_id))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(mainwin, "Run failed", str(exc))

    def _stop_service() -> None:
        n = _selected_service_container()
        if n is None:
            return
        proc_mgr.stop(_service_id_for_container(n))

    def _deploy_selected() -> None:
        n = _selected_service_container()
        if n is None:
            return
        service_id = _service_id_for_container(n)
        try:
            nats_url = "nats://127.0.0.1:4222"
            nodes = []
            try:
                nodes = list(getattr(n, "contained_nodes")() or [])
            except Exception:
                nodes = []
            rt = export_runtime_graph(graph, service_id=service_id, include_nodes=nodes)
            asyncio.run(deploy_to_service(service_id=service_id, nats_url=nats_url, graph=rt))
            QtWidgets.QMessageBox.information(mainwin, "Deploy", f"Deployed to {service_id}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(mainwin, "Deploy failed", str(exc))

    def _deploy_and_run() -> None:
        _run_service()
        _deploy_selected()

    menu = mainwin.menuBar().addMenu("Graph")
    menu.addAction(deploy_action)
    menu.addSeparator()
    menu.addAction(QtWidgets.QAction("Wrap Selected Into Engine", mainwin, triggered=_wrap_selected_into_container))
    menu.addAction(QtWidgets.QAction("Run Engine", mainwin, triggered=_run_service))
    menu.addAction(QtWidgets.QAction("Stop Engine", mainwin, triggered=_stop_service))
    menu.addAction(QtWidgets.QAction("Deploy Selected Engine", mainwin, triggered=_deploy_selected))
    menu.addAction(QtWidgets.QAction("Deploy + Run", mainwin, triggered=_deploy_and_run))

    app.exec_()


if __name__ == "__main__":
    _main()
