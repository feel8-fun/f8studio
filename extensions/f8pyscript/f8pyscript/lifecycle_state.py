from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PyScriptLifecycleState:
    started: bool = False
    paused: bool = False
    active: bool = True
    closing: bool = False

    @property
    def should_stop_in_destructor(self) -> bool:
        return bool(self.started and not self.closing)

    @property
    def can_read_video_latest(self) -> bool:
        return bool(self.active and not self.paused)

    @property
    def can_tick(self) -> bool:
        return bool(self.started and not self.paused)

    @property
    def can_handle_data(self) -> bool:
        return bool(self.active and not self.paused)

    def mark_started(self) -> None:
        self.started = True
        self.paused = False

    def mark_compile_failed(self) -> None:
        self.started = False

    def begin_close(self) -> bool:
        if self.closing:
            return False
        self.closing = True
        return True

    def mark_stopped(self) -> None:
        self.started = False
        self.paused = False

    def set_active(self, active: bool) -> None:
        self.active = bool(active)

    def mark_paused(self) -> None:
        self.paused = True

    def mark_resumed(self) -> None:
        self.paused = False


__all__ = ["PyScriptLifecycleState"]
