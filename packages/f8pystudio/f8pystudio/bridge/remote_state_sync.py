from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..remote_state_watcher import RemoteStateWatcher, WatchTarget


@dataclass(frozen=True)
class ApplyWatchTargetsRequest:
    targets: tuple[WatchTarget, ...]


class RemoteStateGateway(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def apply_targets(self, req: ApplyWatchTargetsRequest) -> None: ...


@dataclass(frozen=True)
class RemoteStateGatewayAdapter:
    watcher: RemoteStateWatcher

    async def start(self) -> None:
        await self.watcher.start()

    async def stop(self) -> None:
        await self.watcher.stop()

    async def apply_targets(self, req: ApplyWatchTargetsRequest) -> None:
        await self.watcher.apply_targets(list(req.targets))
