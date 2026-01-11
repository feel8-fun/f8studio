from NodeGraphQt import NodeObject, BaseNode
from NodeGraphQt.base.node import _ClassProperty

from f8pysdk import F8OperatorSpec, F8ServiceSpec


class GenericRenderNode(BaseNode):

    spec: F8OperatorSpec | F8ServiceSpec

    def __init__(self, qgraphics_item=None):
        super().__init__(qgraphics_item=qgraphics_item)