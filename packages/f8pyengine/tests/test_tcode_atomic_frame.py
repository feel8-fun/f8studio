from __future__ import annotations

from types import MethodType
from typing import Any

import pytest

from f8pyengine.constants import SERVICE_CLASS
from f8pyengine.operators.tcode import TCodeRuntimeNode
from f8pysdk.specs import F8RuntimeNode


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_atomic_frame_avoids_six_scalar_pulls() -> None:
    spec = TCodeRuntimeNode.SPEC
    descriptor = F8RuntimeNode(
        nodeId="tcode",
        serviceId="svcA",
        serviceClass=SERVICE_CLASS,
        operatorClass=spec.operatorClass,
        dataInPorts=list(spec.dataInPorts or []),
        dataOutPorts=list(spec.dataOutPorts or []),
        stateFields=list(spec.stateFields or []),
        stateValues={"intervalMs": 20},
    )
    node = TCodeRuntimeNode(node_id="tcode", node=descriptor, initial_state={"intervalMs": 20})
    pulls: list[str] = []

    async def pull(self: Any, port: str, *, ctx_id: str | int | None = None) -> Any:
        del self, ctx_id
        pulls.append(port)
        if port == "frame":
            return {"axes": {"L0": 0.1, "L1": 0.2, "L2": 0.3, "R0": 0.4, "R1": 0.5, "R2": 0.6}}
        if port == "intervalMs":
            return 20
        raise AssertionError(f"unexpected scalar pull: {port}")

    node.pull = MethodType(pull, node)

    output = await node.compute_output("tcode", ctx_id="frame-1")

    assert output == "L01000I020 L12000I020 L23000I020 R04000I020 R15000I020 R25999I020\n"
    assert pulls == ["intervalMs", "frame"]
