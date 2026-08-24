from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

ChildProcess = subprocess.Popen[bytes]


@dataclass
class _SupervisorStopState:
    requested: bool = False


if os.name == "nt":
    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _KERNEL32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _KERNEL32.AssignProcessToJobObject.restype = wintypes.BOOL
    _KERNEL32.CloseHandle.argtypes = [wintypes.HANDLE]
    _KERNEL32.CloseHandle.restype = wintypes.BOOL
    _KERNEL32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _KERNEL32.CreateJobObjectW.restype = wintypes.HANDLE
    _KERNEL32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _KERNEL32.OpenProcess.restype = wintypes.HANDLE
    _KERNEL32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _KERNEL32.SetInformationJobObject.restype = wintypes.BOOL
else:
    _KERNEL32 = None


def _windows_error_message() -> str:
    code = ctypes.get_last_error()
    return f"WinError {int(code)}"


def _close_windows_handle(handle: object) -> None:
    if os.name != "nt" or _KERNEL32 is None:
        return
    try:
        _KERNEL32.CloseHandle(handle)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[supervisor] CloseHandle failed: {type(exc).__name__}: {exc}", file=sys.stderr)


if os.name == "nt":
    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]


    class _JobObjectBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]


    class _JobObjectExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobObjectBasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


def _create_windows_kill_on_close_job(proc: ChildProcess) -> object | None:
    if os.name != "nt" or _KERNEL32 is None:
        return None
    job = _KERNEL32.CreateJobObjectW(None, None)
    if not job:
        print(f"[supervisor] CreateJobObjectW failed: {_windows_error_message()}", file=sys.stderr)
        return None
    info = _JobObjectExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = _KERNEL32.SetInformationJobObject(
        job,
        9,  # JobObjectExtendedLimitInformation
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        print(f"[supervisor] SetInformationJobObject failed: {_windows_error_message()}", file=sys.stderr)
        _close_windows_handle(job)
        return None
    process = _KERNEL32.OpenProcess(
        0x0001 | 0x0100,  # PROCESS_TERMINATE | PROCESS_SET_QUOTA
        False,
        int(proc.pid),
    )
    if not process:
        print(f"[supervisor] OpenProcess for child job failed pid={proc.pid}: {_windows_error_message()}", file=sys.stderr)
        _close_windows_handle(job)
        return None
    try:
        ok = _KERNEL32.AssignProcessToJobObject(job, process)
    finally:
        _close_windows_handle(process)
    if not ok:
        print(f"[supervisor] AssignProcessToJobObject failed pid={proc.pid}: {_windows_error_message()}", file=sys.stderr)
        _close_windows_handle(job)
        return None
    return job


def _parent_alive(parent_pid: int) -> bool:
    if parent_pid <= 0:
        return False
    if os.name == "nt":
        if _KERNEL32 is None:
            return False
        handle = _KERNEL32.OpenProcess(0x1000, False, int(parent_pid))
        if not handle:
            return False
        _close_windows_handle(handle)
        return True
    try:
        os.kill(int(parent_pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _kill_process_tree(proc: ChildProcess) -> None:
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(int(proc.pid)), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                check=False,
            )
        except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
            print(f"[supervisor] taskkill failed pid={proc.pid}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError as exc:
        print(f"[supervisor] killpg failed pid={proc.pid}: {type(exc).__name__}: {exc}", file=sys.stderr)
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        print(f"[supervisor] child did not exit after kill pid={proc.pid}", file=sys.stderr)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[supervisor] wait after kill failed pid={proc.pid}: {type(exc).__name__}: {exc}", file=sys.stderr)


def _terminate_process_tree(proc: ChildProcess, *, grace_s: float) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            proc.terminate()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"[supervisor] terminate failed pid={proc.pid}: {type(exc).__name__}: {exc}", file=sys.stderr)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError as exc:
            print(f"[supervisor] terminate process group failed pid={proc.pid}: {type(exc).__name__}: {exc}", file=sys.stderr)
    try:
        proc.wait(timeout=max(0.0, float(grace_s)))
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)


def _wait_then_terminate_process_tree(proc: ChildProcess, *, wait_s: float, terminate_grace_s: float) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.wait(timeout=max(0.0, float(wait_s)))
        return
    except subprocess.TimeoutExpired:
        pass
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[supervisor] wait before terminate failed pid={proc.pid}: {type(exc).__name__}: {exc}", file=sys.stderr)
    _terminate_process_tree(proc, grace_s=float(terminate_grace_s))


def _install_signal_handlers(
    proc: ChildProcess,
    *,
    grace_s: float,
    soft_wait_s: float,
    stop_state: _SupervisorStopState,
) -> None:
    stopping = threading.Event()

    def _handle_gentle_signal(signum: int, _frame: object) -> None:
        if stopping.is_set():
            return
        stopping.set()
        stop_state.requested = True
        print(f"[supervisor] signal {int(signum)} received; waiting before terminate child pid={proc.pid}", file=sys.stderr)
        _wait_then_terminate_process_tree(proc, wait_s=float(soft_wait_s), terminate_grace_s=float(grace_s))

    def _handle_terminate_signal(signum: int, _frame: object) -> None:
        if stopping.is_set():
            return
        stopping.set()
        stop_state.requested = True
        print(f"[supervisor] signal {int(signum)} received; terminate child pid={proc.pid}", file=sys.stderr)
        _terminate_process_tree(proc, grace_s=float(grace_s))

    try:
        signal.signal(signal.SIGINT, _handle_gentle_signal)
    except (OSError, RuntimeError, ValueError):
        pass
    try:
        signal.signal(signal.SIGTERM, _handle_terminate_signal)
    except (OSError, RuntimeError, ValueError):
        pass
    if os.name == "nt":
        try:
            signal.signal(signal.SIGBREAK, _handle_gentle_signal)
        except (AttributeError, OSError, RuntimeError, ValueError):
            pass


def _start_stdin_control_thread(
    proc: ChildProcess,
    *,
    grace_s: float,
    soft_wait_s: float,
    stop_state: _SupervisorStopState,
) -> None:
    stopping = threading.Event()

    def _run() -> None:
        try:
            for raw_line in sys.stdin:
                command = str(raw_line or "").strip().lower()
                if command not in {"stop", "quit", "exit"}:
                    continue
                if stopping.is_set():
                    return
                stopping.set()
                stop_state.requested = True
                print(f"[supervisor] stdin stop requested; waiting for child pid={proc.pid}", file=sys.stderr)
                _wait_then_terminate_process_tree(proc, wait_s=float(soft_wait_s), terminate_grace_s=float(grace_s))
                return
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"[supervisor] stdin control failed pid={proc.pid}: {type(exc).__name__}: {exc}", file=sys.stderr)

    thread = threading.Thread(target=_run, name="supervisor-stdin-control", daemon=True)
    thread.start()


def _run_supervisor(
    *,
    parent_pid: int,
    poll_s: float,
    grace_s: float,
    soft_wait_s: float,
    child_cmd: Sequence[str],
) -> int:
    if not child_cmd:
        raise ValueError("child command is empty")
    popen_kwargs: dict[str, object] = {"stdin": subprocess.DEVNULL}
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    else:
        try:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        except AttributeError:
            pass
    proc = subprocess.Popen(list(child_cmd), **popen_kwargs)
    job = _create_windows_kill_on_close_job(proc)
    print(f"[supervisor] child started pid={proc.pid} parentPid={int(parent_pid)}", flush=True)
    try:
        stop_state = _SupervisorStopState()
        _install_signal_handlers(
            proc,
            grace_s=float(grace_s),
            soft_wait_s=float(soft_wait_s),
            stop_state=stop_state,
        )
        _start_stdin_control_thread(
            proc,
            grace_s=float(grace_s),
            soft_wait_s=float(soft_wait_s),
            stop_state=stop_state,
        )
        while True:
            rc = proc.poll()
            if rc is not None:
                if stop_state.requested:
                    return 0
                if int(rc) != 0:
                    rc_int = int(rc)
                    if rc_int < 0:
                        rc_hex = f"-0x{abs(rc_int):X}"
                    else:
                        rc_hex = f"0x{rc_int:X}"
                    hint = ""
                    if rc_int == 0xC0000135:
                        hint = " hint=Windows loader failed to find a required DLL"
                    print(
                        f"[supervisor] child exited rc={rc_int} ({rc_hex}){hint} pid={proc.pid} cmd={' '.join(child_cmd)}",
                        file=sys.stderr,
                        flush=True,
                    )
                return int(rc)
            if not _parent_alive(int(parent_pid)):
                print(f"[supervisor] parent pid={int(parent_pid)} disappeared; stopping child pid={proc.pid}", file=sys.stderr)
                _terminate_process_tree(proc, grace_s=float(grace_s))
                return 0
            time.sleep(max(0.05, float(poll_s)))
    finally:
        if job is not None:
            _close_windows_handle(job)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a service child bound to the PyStudio parent process.")
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--poll-s", default=0.5, type=float)
    parser.add_argument("--soft-wait-s", default=2.0, type=float)
    parser.add_argument("--grace-s", default=2.0, type=float)
    parser.add_argument("child_cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv) if argv is not None else None)
    child_cmd = list(args.child_cmd)
    if child_cmd and child_cmd[0] == "--":
        child_cmd = child_cmd[1:]
    return _run_supervisor(
        parent_pid=int(args.parent_pid),
        poll_s=float(args.poll_s),
        grace_s=float(args.grace_s),
        soft_wait_s=float(args.soft_wait_s),
        child_cmd=child_cmd,
    )


if __name__ == "__main__":
    raise SystemExit(main())
