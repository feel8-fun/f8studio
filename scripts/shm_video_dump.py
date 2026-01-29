import argparse
import os
import sys
import time

try:
    from f8pysdk.shm import VideoShmReader, default_video_shm_name
except ModuleNotFoundError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(repo_root, "packages", "f8pysdk"))
    from f8pysdk.shm import VideoShmReader, default_video_shm_name


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

    shm_name = default_video_shm_name(args.service_id)
    started = time.time()
    reader: VideoShmReader | None = None
    while time.time() - started < args.timeout:
        try:
            reader = VideoShmReader(shm_name)
            reader.open(use_event=False)
            break
        except FileNotFoundError:
            time.sleep(0.1)
    if reader is None:
        print(f"[shm] not found name={shm_name}", file=sys.stderr)
        return 3

    os.makedirs(args.out_dir, exist_ok=True)
    dumped = 0
    last_id = 0
    try:
        while time.time() - started < args.timeout and dumped < int(args.frames):
            hdr = reader.read_header()
            if not hdr:
                time.sleep(0.02)
                continue
            if hdr.width <= 0 or hdr.height <= 0 or hdr.pitch <= 0 or hdr.payload_capacity <= 0:
                time.sleep(0.02)
                continue
            if hdr.frame_id == last_id:
                time.sleep(0.02)
                continue

            last_id = hdr.frame_id
            hdr2, frame_view = reader.read_latest_bgra()
            if not hdr2 or frame_view is None:
                time.sleep(0.02)
                continue
            raw = bytes(frame_view)
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
            if reader:
                reader.close()
        except Exception:
            pass

    if dumped < int(args.frames):
        print(f"[shm] timeout; dumped={dumped}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
