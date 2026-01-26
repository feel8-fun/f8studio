import argparse
import importlib
import os
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

from multiprocessing.shared_memory import SharedMemory


def _require(module_name: str, pip_name: Optional[str] = None):
    try:
        return importlib.import_module(module_name)
    except Exception:
        pkg = pip_name or module_name
        print(f"Missing Python dependency: {module_name}", file=sys.stderr)
        print(f"Install: python -m pip install {pkg}", file=sys.stderr)
        raise


AUDIO_SHM_MAGIC = 0xF8A11A02
AUDIO_SHM_VERSION = 1

SAMPLE_FORMAT_F32LE = 1
SAMPLE_FORMAT_S16LE = 2

_AUDIO_HEADER_STRUCT = struct.Struct("<IIIHHIIIIQQq")
_CHUNK_HEADER_STRUCT = struct.Struct("<QqII")


@dataclass(frozen=True)
class AudioShmHeader:
    magic: int
    version: int
    sample_rate: int
    channels: int
    fmt: int
    frames_per_chunk: int
    chunk_count: int
    bytes_per_frame: int
    payload_bytes_per_chunk: int
    write_seq: int
    write_frame_index: int
    ts_ms: int

    @property
    def header_bytes(self) -> int:
        return _AUDIO_HEADER_STRUCT.size

    @property
    def chunk_header_bytes(self) -> int:
        return _CHUNK_HEADER_STRUCT.size

    @property
    def chunk_stride_bytes(self) -> int:
        return self.chunk_header_bytes + int(self.payload_bytes_per_chunk)


@dataclass(frozen=True)
class AudioChunkHeader:
    seq: int
    ts_ms: int
    frames: int


def compute_default_audio_shm_name(service_id: str) -> str:
    return f"shm.{service_id}.audio"


def read_audio_header(buf: memoryview) -> Optional[AudioShmHeader]:
    if len(buf) < _AUDIO_HEADER_STRUCT.size:
        return None
    try:
        fields = _AUDIO_HEADER_STRUCT.unpack_from(buf, 0)
    except Exception:
        return None
    return AudioShmHeader(
        magic=fields[0],
        version=fields[1],
        sample_rate=fields[2],
        channels=fields[3],
        fmt=fields[4],
        frames_per_chunk=fields[5],
        chunk_count=fields[6],
        bytes_per_frame=fields[7],
        payload_bytes_per_chunk=fields[8],
        write_seq=fields[9],
        write_frame_index=fields[10],
        ts_ms=fields[11],
    )


def read_chunk_header(buf: memoryview, offset: int) -> Optional[AudioChunkHeader]:
    if offset < 0 or offset + _CHUNK_HEADER_STRUCT.size > len(buf):
        return None
    try:
        fields = _CHUNK_HEADER_STRUCT.unpack_from(buf, offset)
    except Exception:
        return None
    return AudioChunkHeader(seq=fields[0], ts_ms=fields[1], frames=fields[2])


class _Win32EventWaiter:
    def __init__(self, event_handle, wait_fn):
        self._event_handle = event_handle
        self._wait_fn = wait_fn
        self._stop = threading.Event()
        self._pending = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="audioshm_event_waiter", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def pop_pending(self) -> int:
        with self._lock:
            v = self._pending
            self._pending = 0
            return v

    def _run(self) -> None:
        WAIT_OBJECT_0 = 0x00000000
        while not self._stop.is_set():
            rc = self._wait_fn(self._event_handle, 200)
            if rc == WAIT_OBJECT_0:
                with self._lock:
                    self._pending += 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize f8 AudioSHM (ring buffer) as a waveform.")
    ap.add_argument("--shm", default="", help="Shared memory mapping name (e.g. shm.audiocap.audio)")
    ap.add_argument("--service-id", default="", help="If set, uses shm.<service-id>.audio")
    ap.add_argument("--use-event", action="store_true", help="Wait on Windows named event shmName_evt when available")
    ap.add_argument("--poll-ms", type=int, default=5, help="Polling interval when no new chunk (ms)")
    ap.add_argument("--history-ms", type=int, default=250, help="Waveform window length (ms)")
    ap.add_argument("--channel", type=int, default=-1, help="Channel to show (0..N-1), default=-1 shows all")
    ap.add_argument("--ylim", type=float, default=0.25, help="Y axis limit (symmetric) for float32")
    ap.add_argument("--title", default="AudioSHM Viewer", help="Window title")
    ap.add_argument("--update-ms", type=int, default=10, help="GUI update interval (ms)")
    ap.add_argument("--prefill", action="store_true", help="Prefill waveform window from recent chunks on first update")
    args = ap.parse_args()

    shm_name = args.shm.strip()
    if not shm_name and args.service_id:
        shm_name = compute_default_audio_shm_name(args.service_id.strip())
    if not shm_name:
        ap.error("Missing --shm or --service-id")

    numpy = _require("numpy")
    pg = _require("pyqtgraph", "pyqtgraph")
    QtCore = pg.Qt.QtCore
    QtWidgets = pg.Qt.QtWidgets

    shm = SharedMemory(name=shm_name, create=False)
    try:
        buf = shm.buf
        print(f"[audioshm] name={shm_name} bytes={shm.size}")

        event_handle = None
        event_waiter = None
        if args.use_event and os.name == "nt":
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            SYNCHRONIZE = 0x00100000

            OpenEventW = kernel32.OpenEventW
            OpenEventW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
            OpenEventW.restype = ctypes.c_void_p

            WaitForSingleObject = kernel32.WaitForSingleObject
            WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            WaitForSingleObject.restype = ctypes.c_uint32

            CloseHandle = kernel32.CloseHandle
            CloseHandle.argtypes = [ctypes.c_void_p]
            CloseHandle.restype = ctypes.c_int

            ev_name = shm_name + "_evt"
            h = OpenEventW(SYNCHRONIZE, 0, ev_name)
            if h:
                event_handle = h
                print(f"[audioshm] using event={ev_name}")
                event_waiter = _Win32EventWaiter(event_handle, WaitForSingleObject)
                event_waiter.start()
            else:
                err = ctypes.get_last_error()
                print(f"[audioshm] event not available ({ev_name}) err={err}, falling back to polling")

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        win = pg.GraphicsLayoutWidget(title=args.title)
        win.resize(1000, 500)
        plot = win.addPlot(row=0, col=0, title=args.title)
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setLabel("bottom", "time", units="ms")
        plot.setLabel("left", "amplitude")
        plot.setYRange(-abs(args.ylim), abs(args.ylim), padding=0.0)

        lines = []
        x = None
        ybuf = None  # shape (channels, window_frames)
        window_frames = 0
        filled_frames = 0
        last_seq = 0
        last_header = None
        chunks_seen = 0
        chunks_start_ms = int(time.time() * 1000)

        def rebuild_plot(new_hdr: AudioShmHeader) -> None:
            nonlocal x, ybuf, window_frames, lines, filled_frames
            window_frames = max(1, int(new_hdr.sample_rate * max(1, args.history_ms) / 1000.0))
            x = numpy.linspace(-args.history_ms, 0.0, num=window_frames, endpoint=False, dtype=numpy.float32)
            chn = int(new_hdr.channels)
            ybuf = numpy.zeros((chn, window_frames), dtype=numpy.float32)
            filled_frames = 0

            plot.clear()
            plot.setTitle(f"{args.title}  {new_hdr.sample_rate}Hz  ch={chn}  chunk={new_hdr.frames_per_chunk} frames")

            lines = []
            if args.channel >= 0:
                safe_ch = int(args.channel)
                if safe_ch < 0 or safe_ch >= chn:
                    safe_ch = 0
                pen = pg.mkPen(width=1)
                line = plot.plot(x, ybuf[safe_ch], pen=pen, name=f"ch{safe_ch}")
                lines.append((safe_ch, line))
            else:
                colors = [pg.intColor(i, hues=chn) for i in range(chn)]
                for ch in range(chn):
                    pen = pg.mkPen(colors[ch], width=1)
                    line = plot.plot(x, ybuf[ch], pen=pen, name=f"ch{ch}")
                    lines.append((ch, line))
            plot.addLegend(offset=(10, 10))

        def read_chunk_samples(new_hdr: AudioShmHeader, seq: int):
            idx = seq % int(new_hdr.chunk_count)
            chunk_base = new_hdr.header_bytes + idx * new_hdr.chunk_stride_bytes
            ch0 = read_chunk_header(buf, chunk_base)
            if ch0 is None or ch0.seq != seq:
                return None, None
            frames = int(min(ch0.frames, new_hdr.frames_per_chunk))
            if frames <= 0:
                return numpy.zeros((int(new_hdr.channels), 0), dtype=numpy.float32), ch0
            payload_off = chunk_base + new_hdr.chunk_header_bytes
            payload_len = frames * int(new_hdr.bytes_per_frame)
            if payload_off + payload_len > len(buf):
                return None, None
            payload = buf[payload_off : payload_off + payload_len]
            samples = numpy.frombuffer(payload, dtype=numpy.float32).copy()
            return samples.reshape((frames, int(new_hdr.channels))).T, ch0

        def prefill_window(new_hdr: AudioShmHeader, seq: int) -> None:
            nonlocal filled_frames
            if ybuf is None:
                return
            frames_per_chunk = int(new_hdr.frames_per_chunk)
            if frames_per_chunk <= 0:
                return
            needed_chunks = max(1, int((window_frames + frames_per_chunk - 1) / frames_per_chunk))
            start_seq = max(1, seq - needed_chunks + 1)

            tmp = None
            for s in range(start_seq, seq + 1):
                smp, _ = read_chunk_samples(new_hdr, s)
                if smp is None or smp.shape[1] == 0:
                    continue
                tmp = smp if tmp is None else numpy.concatenate((tmp, smp), axis=1)

            if tmp is None or tmp.size == 0:
                return
            tmp = tmp[:, -window_frames:]
            ybuf[:, :] = 0.0
            ybuf[:, -tmp.shape[1] :] = tmp
            filled_frames = min(window_frames, int(tmp.shape[1]))

        def update_once() -> None:
            nonlocal last_seq, last_header, chunks_seen, chunks_start_ms, filled_frames
            hdr = read_audio_header(buf)
            if hdr is None or hdr.magic != AUDIO_SHM_MAGIC or hdr.version != AUDIO_SHM_VERSION:
                return
            if hdr.fmt != SAMPLE_FORMAT_F32LE:
                return
            if hdr.sample_rate <= 0 or hdr.channels <= 0 or hdr.frames_per_chunk <= 0 or hdr.chunk_count <= 0:
                return

            key = (
                hdr.sample_rate,
                hdr.channels,
                hdr.frames_per_chunk,
                hdr.chunk_count,
                hdr.bytes_per_frame,
                hdr.payload_bytes_per_chunk,
            )
            if last_header != key:
                last_header = key
                rebuild_plot(hdr)

            if event_waiter and hdr.write_seq == last_seq and event_waiter.pop_pending() == 0:
                return
            if hdr.write_seq == last_seq:
                return

            seq = int(hdr.write_seq)
            if args.prefill and filled_frames == 0 and window_frames > 0:
                prefill_window(hdr, seq)

            samples, ch0 = read_chunk_samples(hdr, seq)
            if samples is None or ch0 is None:
                return
            frames = int(samples.shape[1])
            last_seq = seq
            if frames <= 0 or ybuf is None:
                return

            chunks_seen += 1

            if frames >= window_frames:
                ybuf[:, :] = samples[:, -window_frames:]
                filled_frames = window_frames
            else:
                ybuf[:, :-frames] = ybuf[:, frames:]
                ybuf[:, -frames:] = samples[:, :frames]
                filled_frames = min(window_frames, filled_frames + frames)

            for ch, line in lines:
                if ch < 0 or ch >= int(hdr.channels):
                    continue
                line.setData(x, ybuf[ch])

            now_ms = int(time.time() * 1000)
            elapsed_s = max(0.001, (now_ms - chunks_start_ms) / 1000.0)
            cps = chunks_seen / elapsed_s
            peak = float(numpy.max(numpy.abs(samples))) if samples.size else 0.0
            plot.setTitle(
                f"{args.title}  {hdr.sample_rate}Hz  ch={hdr.channels}  chunk={hdr.frames_per_chunk} frames"
                f"   seq={seq}   chunks/s={cps:.1f}   peak={peak:.3f}   ts={ch0.ts_ms}"
            )

        timer = QtCore.QTimer()
        timer.setInterval(max(int(args.update_ms), 1))
        timer.timeout.connect(update_once)
        timer.start()

        win.show()
        app.exec()

    finally:
        if "event_waiter" in locals() and event_waiter:
            try:
                event_waiter.stop()
            except Exception:
                pass
        if "event_handle" in locals() and event_handle:
            try:
                import ctypes

                ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(event_handle)
            except Exception:
                pass
        try:
            shm.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
