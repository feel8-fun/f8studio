from __future__ import annotations

import ctypes
import logging
import os
import platform
import queue
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, cast


logger = logging.getLogger(__name__)

WM_HOTKEY = 0x0312
WM_APP = 0x8000
WM_F8_HOTKEY_CONTROL = WM_APP + 1
_REPEATING_ERROR_LOG_INTERVAL_S = 2.0


class NativeHotkeyError(RuntimeError):
    pass


class NativeHotkeyParseError(NativeHotkeyError):
    pass


class NativeHotkeyRegistrationError(NativeHotkeyError):
    pass


class NativeHotkeyUnsupportedError(NativeHotkeyError):
    pass


@dataclass(frozen=True)
class NativeHotkeySpec:
    key_name: str
    ctrl: bool
    alt: bool
    shift: bool
    meta: bool
    display_text: str


@dataclass(frozen=True)
class NativeHotkeyBinding:
    binding_id: str
    spec: NativeHotkeySpec


class NativeHotkeyBackend(Protocol):
    def register_hotkey(self, binding: NativeHotkeyBinding) -> None: ...

    def unregister_all(self) -> None: ...

    def close(self) -> None: ...


_MODIFIER_ALIASES = {
    "alt": "Alt",
    "cmd": "Meta",
    "command": "Meta",
    "control": "Ctrl",
    "ctrl": "Ctrl",
    "meta": "Meta",
    "option": "Alt",
    "shift": "Shift",
    "super": "Meta",
    "win": "Meta",
    "windows": "Meta",
}
_KEY_ALIASES = {
    "backspace": "Backspace",
    "bksp": "Backspace",
    "comma": "Comma",
    "del": "Delete",
    "delete": "Delete",
    "down": "Down",
    "end": "End",
    "enter": "Enter",
    "esc": "Escape",
    "escape": "Escape",
    "equal": "Equal",
    "equals": "Equal",
    "home": "Home",
    "ins": "Insert",
    "insert": "Insert",
    "left": "Left",
    "minus": "Minus",
    "pagedown": "PageDown",
    "pageup": "PageUp",
    "pgdn": "PageDown",
    "pgdown": "PageDown",
    "pgup": "PageUp",
    "period": "Period",
    "plus": "Plus",
    "quote": "Quote",
    "return": "Enter",
    "right": "Right",
    "semicolon": "Semicolon",
    "slash": "Slash",
    "space": "Space",
    "tab": "Tab",
    "up": "Up",
}
_KEY_ALIASES.update({f"f{index}": f"F{index}" for index in range(1, 25)})
_KEY_ALIASES.update({character: character.upper() for character in "abcdefghijklmnopqrstuvwxyz"})
_KEY_ALIASES.update({digit: digit for digit in "0123456789"})


def parse_native_hotkey(text: str) -> NativeHotkeySpec:
    raw_text = text.strip()
    if not raw_text:
        raise NativeHotkeyParseError("Global hotkey cannot be empty")
    tokens = [part.strip() for part in raw_text.split("+")]
    if any(not token for token in tokens):
        raise NativeHotkeyParseError(f"Invalid hotkey format: {raw_text!r}")

    modifiers: set[str] = set()
    key_name = ""
    for token in tokens:
        normalized_token = token.lower()
        modifier = _MODIFIER_ALIASES.get(normalized_token)
        if modifier is not None:
            modifiers.add(modifier)
            continue
        key = _KEY_ALIASES.get(normalized_token)
        if key is None:
            raise NativeHotkeyParseError(f"Unsupported hotkey token: {token!r}")
        if key_name:
            raise NativeHotkeyParseError("Only single-key shortcuts are supported")
        key_name = key
    if not key_name:
        raise NativeHotkeyParseError("Hotkey must include a non-modifier key")
    if not modifiers:
        raise NativeHotkeyParseError("Global hotkey must include at least one modifier")

    ordered = [name for name in ("Ctrl", "Alt", "Shift", "Meta") if name in modifiers]
    return NativeHotkeySpec(
        key_name=key_name,
        ctrl="Ctrl" in modifiers,
        alt="Alt" in modifiers,
        shift="Shift" in modifiers,
        meta="Meta" in modifiers,
        display_text="+".join((*ordered, key_name)),
    )


class _Win32Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _Win32Msg(ctypes.Structure):
    _fields_ = [
        ("hwnd", ctypes.c_void_p),
        ("message", ctypes.c_uint32),
        ("wParam", ctypes.c_size_t),
        ("lParam", ctypes.c_size_t),
        ("time", ctypes.c_uint32),
        ("pt", _Win32Point),
    ]


class _RegisterHotKey(Protocol):
    def __call__(self, hwnd: object | None, hotkey_id: int, modifiers: int, virtual_key: int) -> int: ...


class _UnregisterHotKey(Protocol):
    def __call__(self, hwnd: object | None, hotkey_id: int) -> int: ...


class _PeekMessage(Protocol):
    def __call__(self, message: object, hwnd: object | None, minimum: int, maximum: int, remove: int) -> int: ...


class _GetMessage(Protocol):
    def __call__(self, message: object, hwnd: object | None, minimum: int, maximum: int) -> int: ...


class _PostThreadMessage(Protocol):
    def __call__(self, thread_id: int, message: int, wparam: int, lparam: int) -> int: ...


@dataclass(frozen=True)
class Win32Apis:
    register_hot_key: _RegisterHotKey
    unregister_hot_key: _UnregisterHotKey
    peek_message: _PeekMessage
    get_message: _GetMessage
    post_thread_message: _PostThreadMessage
    get_current_thread_id: Callable[[], int]
    set_last_error: Callable[[int], None]
    get_last_error: Callable[[], int]


def build_win32_apis() -> Win32Apis:
    if platform.system() != "Windows":
        raise NativeHotkeyUnsupportedError("Win32 global hotkeys require Windows")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    return Win32Apis(
        register_hot_key=cast(_RegisterHotKey, user32.RegisterHotKey),
        unregister_hot_key=cast(_UnregisterHotKey, user32.UnregisterHotKey),
        peek_message=cast(_PeekMessage, user32.PeekMessageW),
        get_message=cast(_GetMessage, user32.GetMessageW),
        post_thread_message=cast(_PostThreadMessage, user32.PostThreadMessageW),
        get_current_thread_id=cast(Callable[[], int], kernel32.GetCurrentThreadId),
        set_last_error=_set_win32_last_error,
        get_last_error=ctypes.get_last_error,
    )


def _set_win32_last_error(value: int) -> None:
    ctypes.set_last_error(value)


@dataclass
class _Win32Command:
    kind: str
    binding: NativeHotkeyBinding | None = None
    done: threading.Event | None = None
    result: queue.Queue[BaseException | None] | None = None


class Win32NativeHotkeyBackend:
    def __init__(
        self,
        *,
        activation_callback: Callable[[str], None],
        apis: Win32Apis | None = None,
    ) -> None:
        self._apis = apis or build_win32_apis()
        self._activation_callback = activation_callback
        self._commands: queue.Queue[_Win32Command] = queue.Queue()
        self._ready = threading.Event()
        self._thread_id = 0
        self._next_hotkey_id = 1
        self._ids_by_binding: dict[str, int] = {}
        self._bindings_by_id: dict[int, str] = {}
        self._thread = threading.Thread(target=self._run, name="f8studio-win32-hotkeys", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=2.0)
        if self._thread_id <= 0:
            raise NativeHotkeyUnsupportedError("Windows global hotkey worker failed to initialize")

    def register_hotkey(self, binding: NativeHotkeyBinding) -> None:
        self._execute(_Win32Command(kind="register", binding=binding))

    def unregister_all(self) -> None:
        self._execute(_Win32Command(kind="unregister_all"))

    def close(self) -> None:
        if self._thread.is_alive():
            self._execute(_Win32Command(kind="stop"))
            self._thread.join(timeout=1.0)

    def _execute(self, command: _Win32Command) -> None:
        command.done = threading.Event()
        command.result = queue.Queue(maxsize=1)
        self._commands.put(command)
        if self._thread_id > 0:
            posted = self._apis.post_thread_message(self._thread_id, WM_F8_HOTKEY_CONTROL, 0, 0)
            if not posted:
                raise NativeHotkeyRegistrationError(
                    f"PostThreadMessageW failed error={self._apis.get_last_error()}"
                )
        if not command.done.wait(timeout=2.0):
            raise NativeHotkeyRegistrationError("Windows global hotkey worker did not respond")
        error = command.result.get_nowait()
        if error is not None:
            raise error

    def _run(self) -> None:
        self._thread_id = self._apis.get_current_thread_id()
        initial = _Win32Msg()
        self._apis.peek_message(ctypes.byref(initial), None, 0, 0, 0)
        self._ready.set()
        while True:
            message = _Win32Msg()
            result = self._apis.get_message(ctypes.byref(message), None, 0, 0)
            if result == 0:
                return
            if result < 0:
                time.sleep(0.01)
                continue
            if int(message.message) == WM_HOTKEY:
                binding_id = self._bindings_by_id.get(int(message.wParam))
                if binding_id is not None:
                    self._activation_callback(binding_id)
            elif int(message.message) == WM_F8_HOTKEY_CONTROL and self._drain_commands():
                return

    def _drain_commands(self) -> bool:
        should_stop = False
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return should_stop
            error: BaseException | None = None
            try:
                if command.kind == "register":
                    if command.binding is None:
                        raise NativeHotkeyRegistrationError("Register command is missing a binding")
                    self._register(command.binding)
                elif command.kind == "unregister_all":
                    self._unregister_all()
                elif command.kind == "stop":
                    self._unregister_all()
                    should_stop = True
                else:
                    raise NativeHotkeyRegistrationError(f"Unsupported worker command: {command.kind}")
            except (NativeHotkeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                error = exc
            if command.result is not None:
                command.result.put_nowait(error)
            if command.done is not None:
                command.done.set()

    def _register(self, binding: NativeHotkeyBinding) -> None:
        hotkey_id = self._next_hotkey_id
        self._next_hotkey_id += 1
        self._apis.set_last_error(0)
        registered = self._apis.register_hot_key(
            None,
            hotkey_id,
            win32_modifiers(binding.spec),
            win32_virtual_key(binding.spec),
        )
        if not registered:
            raise NativeHotkeyRegistrationError(
                f"RegisterHotKey failed binding={binding.binding_id!r} "
                f"hotkey={binding.spec.display_text!r} error={self._apis.get_last_error()}"
            )
        self._ids_by_binding[binding.binding_id] = hotkey_id
        self._bindings_by_id[hotkey_id] = binding.binding_id

    def _unregister_all(self) -> None:
        for hotkey_id in tuple(self._bindings_by_id):
            try:
                self._apis.unregister_hot_key(None, hotkey_id)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.exception("Win32 global hotkey unregister failed hotkey_id=%s", hotkey_id, exc_info=exc)
        self._ids_by_binding.clear()
        self._bindings_by_id.clear()


def win32_modifiers(spec: NativeHotkeySpec) -> int:
    result = 0
    if spec.alt:
        result |= 0x0001
    if spec.ctrl:
        result |= 0x0002
    if spec.shift:
        result |= 0x0004
    if spec.meta:
        result |= 0x0008
    return result


def win32_virtual_key(spec: NativeHotkeySpec) -> int:
    key = spec.key_name
    if len(key) == 1 and (key.isalpha() or key.isdigit()):
        return ord(key.upper())
    if key.startswith("F"):
        try:
            index = int(key[1:])
        except ValueError as exc:
            raise NativeHotkeyRegistrationError(f"Unsupported Windows hotkey key: {key!r}") from exc
        if 1 <= index <= 24:
            return 0x70 + index - 1
    special = {
        "Backspace": 0x08, "Delete": 0x2E, "Down": 0x28, "End": 0x23,
        "Enter": 0x0D, "Escape": 0x1B, "Home": 0x24, "Insert": 0x2D,
        "Left": 0x25, "PageDown": 0x22, "PageUp": 0x21, "Right": 0x27,
        "Space": 0x20, "Tab": 0x09, "Up": 0x26, "Comma": 0xBC,
        "Equal": 0xBB, "Minus": 0xBD, "Period": 0xBE, "Plus": 0xBB,
        "Quote": 0xDE, "Semicolon": 0xBA, "Slash": 0xBF,
    }
    value = special.get(key)
    if value is None:
        raise NativeHotkeyRegistrationError(f"Unsupported Windows hotkey key: {key!r}")
    return value


class _XRoot(Protocol):
    def grab_key(self, keycode: int, modifiers: int, owner_events: bool, pointer_mode: int, keyboard_mode: int) -> None: ...

    def ungrab_key(self, keycode: int, modifiers: int) -> None: ...


class _XScreen(Protocol):
    root: _XRoot


class _XEvent(Protocol):
    type: int
    detail: int
    state: int


class _XDisplay(Protocol):
    def screen(self) -> _XScreen: ...

    def keysym_to_keycode(self, keysym: int) -> int: ...

    def get_modifier_mapping(self) -> object: ...

    def pending_events(self) -> int: ...

    def next_event(self) -> _XEvent: ...

    def sync(self) -> None: ...

    def close(self) -> None: ...


class _XModule(Protocol):
    ShiftMask: int
    LockMask: int
    ControlMask: int
    Mod1Mask: int
    Mod2Mask: int
    Mod3Mask: int
    Mod4Mask: int
    Mod5Mask: int
    GrabModeAsync: int
    KeyPress: int


class _XKModule(Protocol):
    def string_to_keysym(self, name: str) -> int: ...


class X11NativeHotkeyBackend:
    _BASE_MODIFIER_MASK = 0xFF

    def __init__(
        self,
        *,
        activation_callback: Callable[[str], None],
        display_factory: Callable[[], _XDisplay] | None = None,
        x_module: _XModule | None = None,
        xk_module: _XKModule | None = None,
        start_listener: bool = True,
    ) -> None:
        if display_factory is None or x_module is None or xk_module is None:
            from Xlib import X, XK
            from Xlib.display import Display

            display_factory = cast(Callable[[], _XDisplay], Display)
            x_module = cast(_XModule, X)
            xk_module = cast(_XKModule, XK)
        self._activation_callback = activation_callback
        self._display = display_factory()
        self._x = x_module
        self._xk = xk_module
        self._root = self._display.screen().root
        self._grabs: dict[str, list[tuple[int, int]]] = {}
        self._event_bindings: dict[tuple[int, int], str] = {}
        self._stop = threading.Event()
        self._last_error_log: dict[str, float] = {}
        self._ignored_masks = self._build_ignored_masks()
        self._listener: threading.Thread | None = None
        if start_listener:
            self._listener = threading.Thread(target=self._event_loop, name="f8studio-x11-hotkeys", daemon=True)
            self._listener.start()

    def register_hotkey(self, binding: NativeHotkeyBinding) -> None:
        keysym = self._xk.string_to_keysym(_x11_keysym_name(binding.spec.key_name))
        keycode = self._display.keysym_to_keycode(keysym)
        if keysym <= 0 or keycode <= 0:
            raise NativeHotkeyRegistrationError(f"Unsupported X11 hotkey key: {binding.spec.key_name!r}")
        base = x11_modifiers(binding.spec, self._x)
        grabs: list[tuple[int, int]] = []
        try:
            for ignored in self._ignored_masks:
                modifiers = base | ignored
                self._root.grab_key(keycode, modifiers, False, self._x.GrabModeAsync, self._x.GrabModeAsync)
                grabs.append((keycode, modifiers))
                self._event_bindings[(keycode, modifiers)] = binding.binding_id
            self._display.sync()
        except Exception as exc:
            for grabbed_keycode, grabbed_modifiers in grabs:
                self._root.ungrab_key(grabbed_keycode, grabbed_modifiers)
                self._event_bindings.pop((grabbed_keycode, grabbed_modifiers), None)
            raise NativeHotkeyRegistrationError(
                f"X11 grab failed binding={binding.binding_id!r} hotkey={binding.spec.display_text!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._grabs[binding.binding_id] = grabs

    def unregister_all(self) -> None:
        for binding_id, grabs in tuple(self._grabs.items()):
            for keycode, modifiers in grabs:
                try:
                    self._root.ungrab_key(keycode, modifiers)
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    self._log_error("ungrab", "X11 hotkey ungrab failed binding=%s", binding_id, exc=exc)
                self._event_bindings.pop((keycode, modifiers), None)
            self._grabs.pop(binding_id, None)
        try:
            self._display.sync()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._log_error("sync", "X11 hotkey sync failed during unregister", exc=exc)

    def close(self) -> None:
        self._stop.set()
        self.unregister_all()
        if self._listener is not None and self._listener.is_alive():
            self._listener.join(timeout=0.2)
        try:
            self._display.close()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._log_error("close", "X11 hotkey display close failed", exc=exc)

    def _event_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self._display.pending_events() <= 0:
                    time.sleep(0.01)
                    continue
                event = self._display.next_event()
                if event.type != self._x.KeyPress:
                    continue
                binding_id = self._event_bindings.get(
                    (event.detail, event.state & self._BASE_MODIFIER_MASK)
                )
                if binding_id is not None:
                    self._activation_callback(binding_id)
            except Exception as exc:
                if self._stop.is_set():
                    return
                self._log_error("event", "X11 hotkey event dispatch failed", exc=exc)
                time.sleep(0.05)

    def _build_ignored_masks(self) -> tuple[int, ...]:
        masks = [0, self._x.LockMask]
        num_lock = self._modifier_mask("Num_Lock")
        if num_lock:
            masks.extend(mask | num_lock for mask in tuple(masks))
        return tuple(sorted(set(masks)))

    def _modifier_mask(self, name: str) -> int:
        keysym = self._xk.string_to_keysym(name)
        keycode = self._display.keysym_to_keycode(keysym)
        if keysym <= 0 or keycode <= 0:
            return 0
        mapping = self._display.get_modifier_mapping()
        if not isinstance(mapping, (tuple, list)):
            return 0
        masks = (
            self._x.ShiftMask, self._x.LockMask, self._x.ControlMask, self._x.Mod1Mask,
            self._x.Mod2Mask, self._x.Mod3Mask, self._x.Mod4Mask, self._x.Mod5Mask,
        )
        for index, raw_entries in enumerate(cast(tuple[object, ...] | list[object], mapping)):
            if index >= len(masks) or not isinstance(raw_entries, Iterable) or isinstance(raw_entries, (str, bytes)):
                continue
            entries = cast(Iterable[object], raw_entries)
            if any(isinstance(entry, int) and entry == keycode for entry in entries):
                return masks[index]
        return 0

    def _log_error(self, key: str, message: str, *args: object, exc: BaseException) -> None:
        now = time.monotonic()
        if now - self._last_error_log.get(key, 0.0) < _REPEATING_ERROR_LOG_INTERVAL_S:
            return
        self._last_error_log[key] = now
        logger.exception(message, *args, exc_info=exc)


def _x11_keysym_name(key: str) -> str:
    return {
        "Backspace": "BackSpace", "Comma": "comma", "Enter": "Return",
        "Equal": "equal", "Minus": "minus", "PageDown": "Next",
        "PageUp": "Prior", "Period": "period", "Plus": "plus",
        "Quote": "apostrophe", "Semicolon": "semicolon", "Slash": "slash",
        "Space": "space",
    }.get(key, key)


def x11_modifiers(spec: NativeHotkeySpec, x_module: _XModule) -> int:
    result = 0
    if spec.shift:
        result |= x_module.ShiftMask
    if spec.ctrl:
        result |= x_module.ControlMask
    if spec.alt:
        result |= x_module.Mod1Mask
    if spec.meta:
        result |= x_module.Mod4Mask
    return result


def create_native_hotkey_backend(
    activation_callback: Callable[[str], None],
    *,
    platform_name: str | None = None,
) -> NativeHotkeyBackend:
    current = (platform_name or platform.system()).strip().lower()
    if current.startswith("win"):
        return Win32NativeHotkeyBackend(activation_callback=activation_callback)
    if current == "linux":
        session_type = os.environ.get("XDG_SESSION_TYPE", "").strip().lower()
        if session_type and session_type != "x11":
            raise NativeHotkeyUnsupportedError(
                f"Linux global hotkeys require X11; current session is {session_type!r}"
            )
        if not os.environ.get("DISPLAY", "").strip():
            raise NativeHotkeyUnsupportedError("Linux global hotkeys require an X11 DISPLAY")
        return X11NativeHotkeyBackend(activation_callback=activation_callback)
    raise NativeHotkeyUnsupportedError(f"Global hotkeys are unsupported on platform {current!r}")


__all__ = [
    "NativeHotkeyBackend",
    "NativeHotkeyBinding",
    "NativeHotkeyError",
    "NativeHotkeyParseError",
    "NativeHotkeyRegistrationError",
    "NativeHotkeySpec",
    "NativeHotkeyUnsupportedError",
    "Win32Apis",
    "Win32NativeHotkeyBackend",
    "X11NativeHotkeyBackend",
    "create_native_hotkey_backend",
    "parse_native_hotkey",
    "win32_modifiers",
    "win32_virtual_key",
    "x11_modifiers",
]
