from __future__ import annotations

from dataclasses import dataclass

from .async_runtime import AsyncRuntimeThread
from .command_client import CommandGateway
from .process_lifecycle import ServiceProcessGateway
from .remote_state_sync import RemoteStateGateway
from .rungraph_deployer import RungraphGateway


@dataclass(frozen=True)
class BridgeFacadeContext:
    async_runtime: AsyncRuntimeThread
    process_gateway: ServiceProcessGateway
    rungraph_gateway: RungraphGateway | None
    remote_state_gateway: RemoteStateGateway | None
    command_gateway: CommandGateway | None
