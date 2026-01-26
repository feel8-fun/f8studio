from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


if os.name == "nt":
    import ctypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _SYNCHRONIZE = 0x00100000
    _EVENT_MODIFY_STATE = 0x0002
    _INFINITE = 0xFFFFFFFF
    _WAIT_OBJECT_0 = 0x00000000

    _OpenEventW = _kernel32.OpenEventW
    _OpenEventW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
    _OpenEventW.restype = ctypes.c_void_p

    _CreateEventW = _kernel32.CreateEventW
    _CreateEventW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p]
    _CreateEventW.restype = ctypes.c_void_p

    _WaitForSingleObject = _kernel32.WaitForSingleObject
    _WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _WaitForSingleObject.restype = ctypes.c_uint32

    _SetEvent = _kernel32.SetEvent
    _SetEvent.argtypes = [ctypes.c_void_p]
    _SetEvent.restype = ctypes.c_int

    _ResetEvent = _kernel32.ResetEvent
    _ResetEvent.argtypes = [ctypes.c_void_p]
    _ResetEvent.restype = ctypes.c_int

    _CloseHandle = _kernel32.CloseHandle
    _CloseHandle.argtypes = [ctypes.c_void_p]
    _CloseHandle.restype = ctypes.c_int


@dataclass
class Win32Event:
    name: str
    _handle: int

    @staticmethod
    def open(name: str) -> Optional["Win32Event"]:
        if os.name != "nt":
            return None
        h = _OpenEventW(_SYNCHRONIZE | _EVENT_MODIFY_STATE, 0, name)
        if not h:
            return None
        return Win32Event(name=name, _handle=int(h))

    @staticmethod
    def create(name: str, manual_reset: bool = True, initial_state: bool = False) -> Optional["Win32Event"]:
        if os.name != "nt":
            return None
        h = _CreateEventW(None, 1 if manual_reset else 0, 1 if initial_state else 0, name)
        if not h:
            return None
        return Win32Event(name=name, _handle=int(h))

    def close(self) -> None:
        if os.name != "nt":
            return
        if getattr(self, "_handle", 0):
            _CloseHandle(self._handle)
            self._handle = 0

    def wait(self, timeout_ms: int) -> bool:
        if os.name != "nt":
            return False
        rc = _WaitForSingleObject(self._handle, int(timeout_ms) if timeout_ms >= 0 else _INFINITE)
        return rc == _WAIT_OBJECT_0

    def pulse(self) -> None:
        if os.name != "nt":
            return
        _SetEvent(self._handle)
        _ResetEvent(self._handle)

