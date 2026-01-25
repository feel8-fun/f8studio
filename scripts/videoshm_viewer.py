import argparse
import os
import struct
import sys
import time
from dataclasses import dataclass
from typing import Optional

from multiprocessing.shared_memory import SharedMemory


def _require(module_name: str, pip_name: Optional[str] = None):
    try:
        return __import__(module_name)
    except Exception:
        pkg = pip_name or module_name
        print(f"Missing Python dependency: {module_name}", file=sys.stderr)
        print(f"Install: python -m pip install {pkg}", file=sys.stderr)
        raise


VIDEO_SHM_MAGIC = 0xF8A11A01
VIDEO_SHM_VERSION = 1
VIDEO_FORMAT_BGRA32 = 1

_VIDEO_HEADER_STRUCT = struct.Struct("<7I4xQq2I")


@dataclass(frozen=True)
class VideoShmHeader:
    magic: int
    version: int
    slot_count: int
    width: int
    height: int
    pitch: int
    fmt: int
    frame_id: int
    ts_ms: int
    active_slot: int
    payload_capacity: int

    @property
    def header_bytes(self) -> int:
        return _VIDEO_HEADER_STRUCT.size

    @property
    def frame_bytes(self) -> int:
        return int(self.pitch) * int(self.height)

    @property
    def slot_offset_bytes(self) -> int:
        return self.header_bytes + int(self.active_slot) * int(self.payload_capacity)


def read_video_header(buf: memoryview) -> Optional[VideoShmHeader]:
    if len(buf) < _VIDEO_HEADER_STRUCT.size:
        return None
    try:
        fields = _VIDEO_HEADER_STRUCT.unpack_from(buf, 0)
    except Exception:
        return None
    return VideoShmHeader(
        magic=fields[0],
        version=fields[1],
        slot_count=fields[2],
        width=fields[3],
        height=fields[4],
        pitch=fields[5],
        fmt=fields[6],
        frame_id=fields[7],
        ts_ms=fields[8],
        active_slot=fields[9],
        payload_capacity=fields[10],
    )


def compute_default_video_shm_name(service_id: str) -> str:
    return f"shm.{service_id}.video"


def main() -> int:
    ap = argparse.ArgumentParser(description="Display frames from f8 video shared memory (BGRA32).")
    ap.add_argument("--shm", default="", help="Shared memory mapping name (e.g. shm.implayer.video)")
    ap.add_argument("--service-id", default="", help="If set, uses shm.<service-id>.video")
    ap.add_argument("--poll-ms", type=int, default=3, help="Polling interval when no new frame (ms)")
    ap.add_argument("--use-event", action="store_true", help="Wait on Windows named event shmName_evt when available")
    ap.add_argument("--max-fps", type=float, default=0.0, help="Limit display FPS (0=unlimited)")
    ap.add_argument("--no-display", action="store_true", help="Only print stats, do not open a window")
    ap.add_argument("--title", default="VideoSHM Viewer", help="Window title")
    args = ap.parse_args()

    shm_name = args.shm.strip()
    if not shm_name and args.service_id:
        shm_name = compute_default_video_shm_name(args.service_id.strip())
    if not shm_name:
        ap.error("Missing --shm or --service-id")

    numpy = _require("numpy")
    cv2 = None if args.no_display else _require("cv2", "opencv-python")

    shm = SharedMemory(name=shm_name, create=False)
    try:
        buf = shm.buf
        print(f"[videoshm] name={shm_name} bytes={shm.size}")

        event_handle = None
        if args.use_event and os.name == "nt":
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            SYNCHRONIZE = 0x00100000
            INFINITE = 0xFFFFFFFF
            WAIT_OBJECT_0 = 0x00000000

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
                print(f"[videoshm] using event={ev_name}")
            else:
                err = ctypes.get_last_error()
                print(f"[videoshm] event not available ({ev_name}) err={err}, falling back to polling")

        last_frame_id = 0
        last_show_ms = 0
        last_stats_ms = 0
        shown_frames = 0
        shown_start_ms = int(time.time() * 1000)

        while True:
            hdr0 = read_video_header(buf)
            if hdr0 is None or hdr0.magic != VIDEO_SHM_MAGIC or hdr0.version != VIDEO_SHM_VERSION:
                time.sleep(max(args.poll_ms, 1) / 1000.0)
                continue
            if hdr0.fmt != VIDEO_FORMAT_BGRA32:
                print(f"[videoshm] unsupported format={hdr0.fmt}, expected={VIDEO_FORMAT_BGRA32}", file=sys.stderr)
                time.sleep(0.25)
                continue
            if hdr0.width == 0 or hdr0.height == 0 or hdr0.pitch == 0 or hdr0.payload_capacity == 0:
                time.sleep(max(args.poll_ms, 1) / 1000.0)
                continue
            if hdr0.frame_bytes > hdr0.payload_capacity:
                print(
                    f"[videoshm] invalid header: frameBytes={hdr0.frame_bytes} > payloadCapacity={hdr0.payload_capacity}",
                    file=sys.stderr,
                )
                time.sleep(0.25)
                continue
            if hdr0.slot_offset_bytes + hdr0.frame_bytes > len(buf):
                print("[videoshm] shm size too small for header/payload", file=sys.stderr)
                time.sleep(0.25)
                continue

            if hdr0.frame_id == last_frame_id:
                now_ms = int(time.time() * 1000)
                if now_ms - last_stats_ms >= 1000:
                    last_stats_ms = now_ms
                    print(
                        f"[videoshm] frameId={hdr0.frame_id} {hdr0.width}x{hdr0.height} pitch={hdr0.pitch} slot={hdr0.active_slot}/{hdr0.slot_count} ts={hdr0.ts_ms}"
                    )
                if event_handle:
                    # Keep UI responsive (OpenCV pumps messages via waitKey).
                    if not args.no_display and cv2:
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break
                    rc = WaitForSingleObject(event_handle, 10)
                    if rc != WAIT_OBJECT_0:
                        time.sleep(max(args.poll_ms, 1) / 1000.0)
                else:
                    time.sleep(max(args.poll_ms, 1) / 1000.0)
                continue

            hdr1 = read_video_header(buf)
            if hdr1 is None:
                continue
            if hdr1.frame_id != hdr0.frame_id or hdr1.active_slot != hdr0.active_slot:
                continue

            frame_view = buf[hdr0.slot_offset_bytes : hdr0.slot_offset_bytes + hdr0.frame_bytes]
            frame = numpy.frombuffer(frame_view, dtype=numpy.uint8).copy()
            frame = frame.reshape((hdr0.height, hdr0.pitch))
            frame = frame[:, : hdr0.width * 4].reshape((hdr0.height, hdr0.width, 4))

            last_frame_id = hdr0.frame_id

            if args.no_display:
                continue

            now_ms = int(time.time() * 1000)
            if args.max_fps > 0:
                min_interval = int(1000.0 / args.max_fps)
                if now_ms - last_show_ms < min_interval:
                    continue
            last_show_ms = now_ms

            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            shown_frames += 1
            elapsed_s = max(0.001, (now_ms - shown_start_ms) / 1000.0)
            fps = shown_frames / elapsed_s
            cv2.putText(
                bgr,
                f"{hdr0.width}x{hdr0.height} frameId={hdr0.frame_id} fps={fps:.1f}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(args.title, bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
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
