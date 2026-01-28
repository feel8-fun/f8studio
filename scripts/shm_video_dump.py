import argparse
import os
import struct
import sys
import time
from dataclasses import dataclass
from typing import Optional


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


def _read_video_header(buf: memoryview) -> Optional[VideoShmHeader]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump webrtc gateway VideoSHM frames to PNG for debugging.")
    parser.add_argument("--service-id", default="webrtc_gateway")
    parser.add_argument("--shm-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to dump.")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--out-dir", default="out/shm_dump")
    args = parser.parse_args()

    try:
        import numpy as np  # type: ignore
        import cv2  # type: ignore
    except Exception as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("Install: uv sync (or pip install numpy opencv-python)", file=sys.stderr)
        return 2

    from multiprocessing.shared_memory import SharedMemory

    shm_name = f"shm.{args.service_id}.video"
    started = time.time()
    shm = None
    while time.time() - started < args.timeout:
        try:
            shm = SharedMemory(name=shm_name, create=False)
            break
        except FileNotFoundError:
            time.sleep(0.1)
    if shm is None:
        print(f"[shm] not found name={shm_name}", file=sys.stderr)
        return 3

    os.makedirs(args.out_dir, exist_ok=True)
    dumped = 0
    last_id = 0
    try:
        buf = shm.buf
        while time.time() - started < args.timeout and dumped < int(args.frames):
            hdr = _read_video_header(buf)
            if not hdr:
                time.sleep(0.02)
                continue
            if hdr.magic != VIDEO_SHM_MAGIC or hdr.version != VIDEO_SHM_VERSION or hdr.fmt != VIDEO_FORMAT_BGRA32:
                time.sleep(0.02)
                continue
            if hdr.width <= 0 or hdr.height <= 0 or hdr.pitch <= 0 or hdr.payload_capacity <= 0:
                time.sleep(0.02)
                continue
            if hdr.frame_id == last_id:
                time.sleep(0.02)
                continue

            last_id = hdr.frame_id
            frame_bytes = hdr.frame_bytes
            off = hdr.slot_offset_bytes
            if off + frame_bytes > len(buf):
                print(f"[shm] invalid frame bounds frameId={hdr.frame_id}", file=sys.stderr)
                time.sleep(0.02)
                continue

            raw = bytes(buf[off : off + frame_bytes])
            img = np.frombuffer(raw, dtype=np.uint8).reshape((hdr.height, hdr.pitch))[:, : hdr.width * 4]
            bgra = img.reshape((hdr.height, hdr.width, 4))
            bgr = bgra[:, :, :3]
            out_path = os.path.join(args.out_dir, f"frame_{hdr.frame_id:06d}.png")
            ok = cv2.imwrite(out_path, bgr)
            print(
                f"[shm] dump frameId={hdr.frame_id} {hdr.width}x{hdr.height} pitch={hdr.pitch} slot={hdr.active_slot}/{hdr.slot_count} ts={hdr.ts_ms} ok={ok} -> {out_path}"
            )
            dumped += 1
    finally:
        try:
            shm.close()
        except Exception:
            pass

    if dumped < int(args.frames):
        print(f"[shm] timeout; dumped={dumped}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

