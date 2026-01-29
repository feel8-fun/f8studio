import argparse
import os
import sys
import time
from typing import Optional

try:
    from f8pysdk.shm import VideoShmReader, default_video_shm_name
except ModuleNotFoundError:
    # Allow running from a source checkout without installing the workspace packages.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(repo_root, "packages", "f8pysdk"))
    from f8pysdk.shm import VideoShmReader, default_video_shm_name


def _require(module_name: str, pip_name: Optional[str] = None):
    try:
        return __import__(module_name)
    except Exception:
        pkg = pip_name or module_name
        print(f"Missing Python dependency: {module_name}", file=sys.stderr)
        print(f"Install: python -m pip install {pkg}", file=sys.stderr)
        raise


def compute_default_video_shm_name(service_id: str) -> str:
    return default_video_shm_name(service_id)


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

    reader = VideoShmReader(shm_name)
    reader.open(use_event=args.use_event)
    try:
        print(f"[videoshm] name={shm_name}")

        last_frame_id = 0
        last_show_ms = 0
        last_stats_ms = 0
        shown_frames = 0
        shown_start_ms = int(time.time() * 1000)

        while True:
            hdr0 = reader.read_header()
            if hdr0 is None or hdr0.width == 0 or hdr0.height == 0 or hdr0.pitch == 0 or hdr0.payload_capacity == 0:
                time.sleep(max(args.poll_ms, 1) / 1000.0)
                continue

            if hdr0.frame_id == last_frame_id:
                now_ms = int(time.time() * 1000)
                if now_ms - last_stats_ms >= 1000:
                    last_stats_ms = now_ms
                    print(
                        f"[videoshm] frameId={hdr0.frame_id} {hdr0.width}x{hdr0.height} pitch={hdr0.pitch} slot={hdr0.active_slot}/{hdr0.slot_count} ts={hdr0.ts_ms}"
                    )
                if not args.no_display and cv2:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                reader.wait_new_frame(timeout_ms=max(1, int(args.poll_ms)))
                continue

            hdr, frame_view = reader.read_latest_bgra()
            if hdr is None or frame_view is None:
                continue
            frame = numpy.frombuffer(frame_view, dtype=numpy.uint8).copy()
            frame = frame.reshape((hdr.height, hdr.pitch))
            frame = frame[:, : hdr.width * 4].reshape((hdr.height, hdr.width, 4))

            last_frame_id = hdr.frame_id

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
                f"{hdr.width}x{hdr.height} frameId={hdr.frame_id} fps={fps:.1f}",
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
        try:
            reader.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
