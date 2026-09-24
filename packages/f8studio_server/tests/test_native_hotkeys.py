from __future__ import annotations

import queue
import threading
import time
from array import array
from pathlib import Path
from typing import Protocol, cast

from f8studio_server.local_integration import LocalIntegrationService, RegisterHotkeyRequest
from f8studio_server.native_hotkeys import (
    NativeHotkeyBinding,
    WM_HOTKEY,
    Win32Apis,
    Win32NativeHotkeyBackend,
    X11NativeHotkeyBackend,
    parse_native_hotkey,
    win32_modifiers,
    win32_virtual_key,
)


class FakeNativeHotkeyBackend:
    def __init__(self) -> None:
        self.registered: list[NativeHotkeyBinding] = []
        self.clear_count = 0
        self.closed = False

    def register_hotkey(self, binding: NativeHotkeyBinding) -> None:
        self.registered.append(binding)

    def unregister_all(self) -> None:
        self.clear_count += 1
        self.registered.clear()

    def close(self) -> None:
        self.closed = True


class _WinMessage(Protocol):
    message: int
    wParam: int
    lParam: int


class _WinMessageHolder(Protocol):
    _obj: _WinMessage


class FakeWin32Apis:
    def __init__(self) -> None:
        self.messages: queue.Queue[tuple[int, int, int]] = queue.Queue()
        self.register_calls: list[tuple[int, int, int]] = []
        self.unregister_calls: list[int] = []
        self.thread_id = 0
        self.last_error = 0

    def register_hot_key(self, _hwnd: object | None, hotkey_id: int, modifiers: int, virtual_key: int) -> int:
        self.register_calls.append((hotkey_id, modifiers, virtual_key))
        return 1

    def unregister_hot_key(self, _hwnd: object | None, hotkey_id: int) -> int:
        self.unregister_calls.append(hotkey_id)
        return 1

    def peek_message(self, _message: object, _hwnd: object | None, _minimum: int, _maximum: int, _remove: int) -> int:
        self.thread_id = threading.get_ident()
        return 0

    def get_message(self, message: object, _hwnd: object | None, _minimum: int, _maximum: int) -> int:
        message_id, wparam, result = self.messages.get(timeout=2.0)
        target = cast(_WinMessageHolder, message)._obj
        target.message = message_id
        target.wParam = wparam
        target.lParam = 0
        return result

    def post_thread_message(self, thread_id: int, message: int, wparam: int, _lparam: int) -> int:
        if thread_id != self.thread_id:
            return 0
        self.messages.put((message, wparam, 1))
        return 1

    @staticmethod
    def get_current_thread_id() -> int:
        return threading.get_ident()

    def set_last_error(self, value: int) -> None:
        self.last_error = value

    def get_last_error(self) -> int:
        return self.last_error

    def emit_hotkey(self, hotkey_id: int) -> None:
        self.messages.put((WM_HOTKEY, hotkey_id, 1))

    def bundle(self) -> Win32Apis:
        return Win32Apis(
            register_hot_key=self.register_hot_key,
            unregister_hot_key=self.unregister_hot_key,
            peek_message=self.peek_message,
            get_message=self.get_message,
            post_thread_message=self.post_thread_message,
            get_current_thread_id=self.get_current_thread_id,
            set_last_error=self.set_last_error,
            get_last_error=self.get_last_error,
        )


class FakeX:
    ShiftMask = 1
    LockMask = 2
    ControlMask = 4
    Mod1Mask = 8
    Mod2Mask = 16
    Mod3Mask = 32
    Mod4Mask = 64
    Mod5Mask = 128
    GrabModeAsync = 1
    KeyPress = 2


class FakeXK:
    @staticmethod
    def string_to_keysym(name: str) -> int:
        return {"P": 42, "Num_Lock": 77}.get(name, 0)


class FakeXRoot:
    def __init__(self) -> None:
        self.grabs: list[tuple[int, int, bool]] = []

    def grab_key(self, keycode: int, modifiers: int, owner_events: bool, _pointer: int, _keyboard: int) -> None:
        self.grabs.append((keycode, modifiers, owner_events))

    def ungrab_key(self, _keycode: int, _modifiers: int) -> None:
        return


class FakeXScreen:
    def __init__(self, root: FakeXRoot) -> None:
        self.root = root


class FakeXDisplay:
    def __init__(self) -> None:
        self.root = FakeXRoot()

    def screen(self) -> FakeXScreen:
        return FakeXScreen(self.root)

    @staticmethod
    def keysym_to_keycode(keysym: int) -> int:
        return {42: 33, 77: 77}.get(keysym, 0)

    @staticmethod
    def get_modifier_mapping() -> list[array[int]]:
        return [array("B"), array("B"), array("B"), array("B"), array("B", [77]), array("B"), array("B"), array("B")]

    @staticmethod
    def pending_events() -> int:
        return 0

    @staticmethod
    def next_event() -> object:
        raise RuntimeError("no fake X11 events")

    @staticmethod
    def sync() -> None:
        return

    @staticmethod
    def close() -> None:
        return


def _request(*, accelerator: str = "Ctrl+Alt+P") -> RegisterHotkeyRequest:
    return RegisterHotkeyRequest(
        accelerator=accelerator,
        project_id="project1",
        node_id="controls",
        field="trigger",
    )


def test_native_hotkey_parser_and_win32_mapping_are_deterministic() -> None:
    spec = parse_native_hotkey("alt + ctrl + f5")

    assert spec.display_text == "Ctrl+Alt+F5"
    assert win32_modifiers(spec) == 0x0001 | 0x0002
    assert win32_virtual_key(spec) == 0x74


def test_win32_worker_registers_dispatches_and_closes_without_qt() -> None:
    fake = FakeWin32Apis()
    activated: list[str] = []
    backend = Win32NativeHotkeyBackend(activation_callback=activated.append, apis=fake.bundle())
    binding = NativeHotkeyBinding(binding_id="project1:controls:trigger", spec=parse_native_hotkey("Ctrl+Alt+P"))

    backend.register_hotkey(binding)
    fake.emit_hotkey(1)
    deadline = time.monotonic() + 1.0
    while not activated and time.monotonic() < deadline:
        time.sleep(0.01)
    backend.close()

    assert activated == [binding.binding_id]
    assert fake.register_calls == [(1, 0x0001 | 0x0002, ord("P"))]
    assert fake.unregister_calls == [1]


def test_x11_registration_handles_array_modifier_mapping_and_focus_outside() -> None:
    display = FakeXDisplay()
    backend = X11NativeHotkeyBackend(
        activation_callback=lambda _binding_id: None,
        display_factory=lambda: display,
        x_module=FakeX(),
        xk_module=FakeXK(),
        start_listener=False,
    )

    backend.register_hotkey(
        NativeHotkeyBinding(binding_id="probe", spec=parse_native_hotkey("Ctrl+Alt+Shift+P"))
    )

    assert display.root.grabs == [
        (33, 13, False),
        (33, 15, False),
        (33, 29, False),
        (33, 31, False),
    ]
    backend.close()


def test_local_hotkeys_register_with_native_backend_and_persist(tmp_path: Path) -> None:
    database_path = tmp_path / "studio.sqlite3"
    backend = FakeNativeHotkeyBackend()
    service = LocalIntegrationService(database_path=database_path, hotkey_backend=backend)

    binding = service.register_hotkey(_request())

    assert binding.status == "registered"
    assert [item.binding_id for item in backend.registered] == [binding.binding_id]
    reopened = LocalIntegrationService(database_path=database_path)
    persisted = reopened.list_hotkeys()
    assert len(persisted) == 1
    assert persisted[0].accelerator == "Ctrl+Alt+P"
    assert persisted[0].node_id == "controls"
    assert persisted[0].field == "trigger"

    service.unregister_hotkey(binding.binding_id)

    assert backend.registered == []
    assert LocalIntegrationService(database_path=database_path).list_hotkeys() == ()


def test_local_hotkey_validator_runs_before_persistence(tmp_path: Path) -> None:
    validated: list[str] = []
    service = LocalIntegrationService(
        database_path=tmp_path / "studio.sqlite3",
        hotkey_validator=lambda binding: validated.append(binding.binding_id),
    )

    binding = service.register_hotkey(_request(accelerator="Ctrl+Shift+K"))

    assert validated == [binding.binding_id]
