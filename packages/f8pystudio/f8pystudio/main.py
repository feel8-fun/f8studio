import argparse
import json
from f8pysdk import F8ServiceDescribe
from f8pystudio.service_host import ServiceHostRegistry
from f8pystudio.service_catalog import load_discovery_into_registries, ServiceCatalog


from qtpy import QtWidgets, QtCore
from NodeGraphQt import (
    NodeGraph,
    NodePropEditorWidget,
    NodesPaletteWidget,
    NodesTreeWidget,
    BaseNode,
    BackdropNode,
    GroupNode,
)


from f8pystudio.widgets.palette_widget import F8NodesPaletteWidget

from f8pystudio.renderNodes import RenderNodeRegistry


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

    palette._category_tabs

    dock = QtWidgets.QDockWidget("Nodes Palette", mainwin)
    dock.setWidget(palette)
    mainwin.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    # dock2 = QtWidgets.QDockWidget("Nodes Tree", mainwin)
    # dock2.setWidget(tree)
    # mainwin.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock2)

    app.exec_()


if __name__ == "__main__":
    _main()
