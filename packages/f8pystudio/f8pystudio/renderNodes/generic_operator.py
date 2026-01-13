from .generic import GenericRenderNode
from f8pysdk import F8OperatorSpec


class GenericOperatorRenderNode(GenericRenderNode):

    spec: F8OperatorSpec
    serviceId: str | None

    def __init__(self, qgraphics_item=None):
        super().__init__(qgraphics_item=qgraphics_item)
        self.serviceId = None
