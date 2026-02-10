import argparse
import importlib
import os
import sys
import time
import traceback
from dataclasses import dataclass
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
        return importlib.import_module(module_name)
    except Exception:
        pkg = pip_name or module_name
        print(f"Missing Python dependency: {module_name}", file=sys.stderr)
        print(f"Install: python -m pip install {pkg}", file=sys.stderr)
        raise


def compute_default_video_shm_name(service_id: str) -> str:
    return default_video_shm_name(service_id)


@dataclass(frozen=True)
class _SaliencyParams:
    threshold: float
    min_area: int
    overlay_alpha: float
    topk: int


@dataclass
class _PerfStats:
    last_print_ms: int
    frames: int = 0
    total_ms: float = 0.0
    shm_ms: float = 0.0
    cvt_ms: float = 0.0
    flow_ms: float = 0.0
    saliency_ms: float = 0.0
    draw_ms: float = 0.0

    def add(
        self,
        *,
        total_ms: float,
        shm_ms: float,
        cvt_ms: float,
        flow_ms: float,
        saliency_ms: float,
        draw_ms: float,
    ) -> None:
        self.frames += 1
        self.total_ms += total_ms
        self.shm_ms += shm_ms
        self.cvt_ms += cvt_ms
        self.flow_ms += flow_ms
        self.saliency_ms += saliency_ms
        self.draw_ms += draw_ms

    def maybe_print(self, now_ms: int) -> None:
        if now_ms - self.last_print_ms < 1000:
            return
        elapsed_ms = max(1, now_ms - self.last_print_ms)
        fps = 1000.0 * float(self.frames) / float(elapsed_ms)
        denom = max(1, self.frames)
        print(
            "[perf] "
            f"fps={fps:.1f} "
            f"total={self.total_ms/denom:.2f}ms "
            f"shm={self.shm_ms/denom:.2f}ms "
            f"cvt={self.cvt_ms/denom:.2f}ms "
            f"flow={self.flow_ms/denom:.2f}ms "
            f"sal={self.saliency_ms/denom:.2f}ms "
            f"draw={self.draw_ms/denom:.2f}ms"
        )
        self.last_print_ms = now_ms
        self.frames = 0
        self.total_ms = 0.0
        self.shm_ms = 0.0
        self.cvt_ms = 0.0
        self.flow_ms = 0.0
        self.saliency_ms = 0.0
        self.draw_ms = 0.0


@dataclass(frozen=True)
class _FlowInfo:
    scale: float
    flow_small: "numpy.ndarray"  # (h, w, 2) in small-pixel units
    med_dx: float
    med_dy: float


def _require_motion_saliency(cv2):
    try:
        saliency_mod = cv2.saliency
    except AttributeError as exc:
        raise RuntimeError(
            "OpenCV Saliency module not found. Install: python -m pip install opencv-contrib-python"
        ) from exc
    try:
        create_fn = saliency_mod.MotionSaliencyBinWangApr2014_create
    except AttributeError as exc:
        raise RuntimeError(
            "MotionSaliencyBinWangApr2014 not found. Install: python -m pip install opencv-contrib-python"
        ) from exc
    return create_fn()


def _compute_motion_saliency(motion_saliency, gray, numpy):
    result = motion_saliency.computeSaliency(gray)
    if isinstance(result, tuple):
        ok, saliency = result
        if not ok:
            return None
        return saliency
    if result is None:
        return None
    if not isinstance(result, numpy.ndarray):
        raise TypeError(f"Unexpected computeSaliency() return type: {type(result)!r}")
    return result


def _compute_flow_info(
    cv2,
    numpy,
    prev_gray: "numpy.ndarray",
    gray: "numpy.ndarray",
    *,
    flow_max_size: int,
) -> _FlowInfo:
    h, w = gray.shape[:2]
    long_side = max(h, w)
    scale = 1.0
    if flow_max_size > 0 and long_side > flow_max_size:
        scale = float(flow_max_size) / float(long_side)

    if scale != 1.0:
        new_w = max(2, int(round(float(w) * scale)))
        new_h = max(2, int(round(float(h) * scale)))
        gray_small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        prev_gray_small = cv2.resize(prev_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        gray_small = gray
        prev_gray_small = prev_gray

    flow_small = cv2.calcOpticalFlowFarneback(
        prev_gray_small,
        gray_small,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    dx = flow_small[..., 0]
    dy = flow_small[..., 1]
    med_dx = float(numpy.median(dx))
    med_dy = float(numpy.median(dy))
    return _FlowInfo(scale=scale, flow_small=flow_small, med_dx=med_dx, med_dy=med_dy)


def _residual_motion_mask_from_flow(
    cv2,
    numpy,
    flow_info: _FlowInfo,
    *,
    full_w: int,
    full_h: int,
    residual_thr_px: float,
    morph_ksize: int,
) -> "numpy.ndarray":
    flow = flow_info.flow_small
    dx = flow[..., 0]
    dy = flow[..., 1]
    resid = numpy.sqrt((dx - flow_info.med_dx) ** 2 + (dy - flow_info.med_dy) ** 2)

    thr_small = float(residual_thr_px) * float(flow_info.scale)
    thr_small = max(0.1, thr_small)
    mask_small = (resid >= thr_small).astype(numpy.uint8) * 255

    if morph_ksize > 1:
        k = int(morph_ksize)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel, iterations=1)

    if flow_info.scale != 1.0:
        mask = cv2.resize(mask_small, (int(full_w), int(full_h)), interpolation=cv2.INTER_NEAREST)
    else:
        mask = mask_small
    return mask


def _draw_saliency(
    cv2,
    numpy,
    bgr,
    saliency: "numpy.ndarray",
    params: _SaliencyParams,
) -> tuple["numpy.ndarray", "numpy.ndarray", list[tuple[int, int, int, int]]]:
    saliency_f32 = saliency.astype(numpy.float32, copy=False)
    saliency_u8 = numpy.clip(saliency_f32 * 255.0, 0.0, 255.0).astype(numpy.uint8)
    saliency_color = cv2.applyColorMap(saliency_u8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(bgr, 1.0 - params.overlay_alpha, saliency_color, params.overlay_alpha, 0.0)

    thr_u8 = int(max(0, min(255, round(params.threshold * 255.0))))
    _, mask = cv2.threshold(saliency_u8, thr_u8, 255, cv2.THRESH_BINARY)

    contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = int(cv2.contourArea(cnt))
        if area >= params.min_area:
            candidates.append((area, cnt))
    candidates.sort(key=lambda x: x[0], reverse=True)

    max_draw = params.topk if params.topk > 0 else len(candidates)
    boxes: list[tuple[int, int, int, int]] = []
    for _area, cnt in candidates[:max_draw]:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        boxes.append((int(x), int(y), int(w), int(h)))

    return overlay, mask, boxes


def _draw_flow_grid(
    cv2,
    numpy,
    img: "numpy.ndarray",
    flow_info: _FlowInfo,
    *,
    step_px: int,
    arrow_scale: float,
    min_mag: float,
    use_residual: bool,
    color: tuple[int, int, int],
) -> None:
    h_full, w_full = img.shape[:2]
    flow = flow_info.flow_small
    h_small, w_small = flow.shape[:2]

    if flow_info.scale <= 0.0:
        return
    inv_scale = 1.0 / float(flow_info.scale)

    step = max(1, int(step_px))
    step_small = max(1, int(round(float(step) * float(flow_info.scale))))

    dx = flow[..., 0].astype(numpy.float32, copy=False)
    dy = flow[..., 1].astype(numpy.float32, copy=False)
    if use_residual:
        dx = dx - float(flow_info.med_dx)
        dy = dy - float(flow_info.med_dy)

    for y in range(0, h_small, step_small):
        yy = int(min(h_full - 1, round(float(y) * inv_scale)))
        for x in range(0, w_small, step_small):
            xx = int(min(w_full - 1, round(float(x) * inv_scale)))
            vx = float(dx[y, x]) * inv_scale
            vy = float(dy[y, x]) * inv_scale
            mag = (vx * vx + vy * vy) ** 0.5
            if mag < float(min_mag):
                continue
            ex = int(round(float(xx) + float(vx) * float(arrow_scale)))
            ey = int(round(float(yy) + float(vy) * float(arrow_scale)))
            ex = max(0, min(w_full - 1, ex))
            ey = max(0, min(h_full - 1, ey))
            cv2.arrowedLine(img, (xx, yy), (ex, ey), color, 1, tipLength=0.25)


def _draw_region_median_flow(
    cv2,
    numpy,
    img: "numpy.ndarray",
    flow_info: _FlowInfo,
    boxes: list[tuple[int, int, int, int]],
    *,
    min_mag: float,
    arrow_scale: float,
    use_residual: bool,
    color: tuple[int, int, int],
) -> None:
    h_full, w_full = img.shape[:2]
    flow = flow_info.flow_small
    h_small, w_small = flow.shape[:2]

    if flow_info.scale <= 0.0:
        return
    scale = float(flow_info.scale)
    inv_scale = 1.0 / scale

    dx = flow[..., 0].astype(numpy.float32, copy=False)
    dy = flow[..., 1].astype(numpy.float32, copy=False)
    if use_residual:
        dx = dx - float(flow_info.med_dx)
        dy = dy - float(flow_info.med_dy)

    for x, y, w, h in boxes:
        cx = int(x + w // 2)
        cy = int(y + h // 2)
        xs0 = int(max(0, min(w_small - 1, int(round(float(x) * scale)))))
        ys0 = int(max(0, min(h_small - 1, int(round(float(y) * scale)))))
        xs1 = int(max(0, min(w_small, int(round(float(x + w) * scale)))))
        ys1 = int(max(0, min(h_small, int(round(float(y + h) * scale)))))
        if xs1 <= xs0 or ys1 <= ys0:
            continue

        roi_dx = dx[ys0:ys1, xs0:xs1]
        roi_dy = dy[ys0:ys1, xs0:xs1]
        if roi_dx.size == 0:
            continue

        med_roi_dx = float(numpy.median(roi_dx))
        med_roi_dy = float(numpy.median(roi_dy))
        vx = med_roi_dx * inv_scale
        vy = med_roi_dy * inv_scale
        mag = (vx * vx + vy * vy) ** 0.5
        if mag < float(min_mag):
            continue
        ex = int(round(float(cx) + float(vx) * float(arrow_scale)))
        ey = int(round(float(cy) + float(vy) * float(arrow_scale)))
        ex = max(0, min(w_full - 1, ex))
        ey = max(0, min(h_full - 1, ey))
        cv2.arrowedLine(img, (cx, cy), (ex, ey), color, 2, tipLength=0.3)


def _put_hud_text(cv2, img, text: str, *, margin: int = 10, base_scale: float = 0.6, thickness: int = 1) -> None:
    h, w = img.shape[:2]
    max_w = max(1, w - 2 * margin)
    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = float(base_scale)
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    if tw > max_w:
        scale = max(0.35, scale * (float(max_w) / float(max(1, tw))))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    x = margin
    y = margin + th
    y = min(y, h - margin)

    pad = 4
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Display f8 VideoSHM with OpenCV Motion Saliency (BinWangApr2014) overlay + bbox."
    )
    ap.add_argument("--shm", default="", help="Shared memory mapping name (e.g. shm.implayer.video)")
    ap.add_argument("--service-id", default="", help="If set, uses shm.<service-id>.video")
    ap.add_argument("--poll-ms", type=int, default=3, help="Polling interval when no new frame (ms)")
    ap.add_argument("--use-event", action="store_true", help="Wait on Windows named event shmName_evt when available")
    ap.add_argument("--max-fps", type=float, default=0.0, help="Limit display FPS (0=unlimited)")
    ap.add_argument("--title", default="VideoSHM Motion Saliency", help="Window title")
    ap.add_argument("--threshold", type=float, default=0.15, help="Saliency threshold in [0..1]")
    ap.add_argument("--min-area", type=int, default=200, help="Min contour area to draw bbox (px)")
    ap.add_argument("--overlay-alpha", type=float, default=0.5, help="Overlay alpha in [0..1]")
    ap.add_argument("--topk", type=int, default=2, help="Keep N largest motion regions (0=all)")
    ap.add_argument(
        "--show-mask",
        action="store_true",
        help="Also show a binary mask window (thresholded saliency)",
    )
    ap.add_argument(
        "--no-suppress-global-motion",
        dest="suppress_global_motion",
        action="store_false",
        help="Disable camera-motion suppression (median optical flow subtraction)",
    )
    ap.set_defaults(suppress_global_motion=True)
    ap.add_argument("--flow-max-size", type=int, default=320, help="Max long side for flow computation (px, 0=full)")
    ap.add_argument("--residual-thr-px", type=float, default=1.5, help="Residual motion threshold in pixels")
    ap.add_argument("--morph-ksize", type=int, default=5, help="Morphology kernel size for motion mask (odd, 0=off)")
    ap.add_argument("--show-flow", action="store_true", help="Draw a downsampled flow grid (arrows)")
    ap.add_argument("--show-region-flow", action="store_true", help="Draw median flow arrow for each bbox region")
    ap.add_argument("--flow-step", type=int, default=24, help="Flow grid spacing (px, in full-res coordinates)")
    ap.add_argument("--flow-arrow-scale", type=float, default=8.0, help="Arrow length multiplier")
    ap.add_argument("--flow-min-mag", type=float, default=0.5, help="Minimum flow magnitude to draw (px)")
    ap.add_argument(
        "--flow-show-raw",
        dest="flow_use_residual",
        action="store_false",
        help="Show raw flow (do not subtract global median motion)",
    )
    ap.set_defaults(flow_use_residual=True)
    ap.add_argument("--print-perf", action="store_true", help="Print per-stage timing once per second")
    args = ap.parse_args()

    shm_name = args.shm.strip()
    if not shm_name and args.service_id:
        shm_name = compute_default_video_shm_name(args.service_id.strip())
    if not shm_name:
        ap.error("Missing --shm or --service-id")

    if args.threshold < 0.0 or args.threshold > 1.0:
        ap.error("--threshold must be in [0..1]")
    if args.overlay_alpha < 0.0 or args.overlay_alpha > 1.0:
        ap.error("--overlay-alpha must be in [0..1]")
    if args.min_area < 0:
        ap.error("--min-area must be >= 0")

    numpy = _require("numpy")
    cv2 = _require("cv2", "opencv-contrib-python")

    params = _SaliencyParams(
        threshold=float(args.threshold),
        min_area=int(args.min_area),
        overlay_alpha=float(args.overlay_alpha),
        topk=int(args.topk),
    )

    motion_saliency = _require_motion_saliency(cv2)

    reader = VideoShmReader(shm_name)
    reader.open(use_event=args.use_event)
    try:
        print(f"[videoshm] name={shm_name}")
        print("[keys] q/esc=quit  m=toggle mask window")

        show_mask = bool(args.show_mask)
        last_frame_id = 0
        last_show_ms = 0
        shown_frames = 0
        shown_start_ms = int(time.time() * 1000)
        configured_size = None  # (w, h)
        prev_gray = None
        perf = _PerfStats(last_print_ms=int(time.time() * 1000))

        while True:
            loop_start = time.perf_counter()
            hdr0 = reader.read_header()
            if hdr0 is None or hdr0.width == 0 or hdr0.height == 0 or hdr0.pitch == 0 or hdr0.payload_capacity == 0:
                time.sleep(max(args.poll_ms, 1) / 1000.0)
                continue

            if hdr0.frame_id == last_frame_id:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("m"),):
                    show_mask = not show_mask
                    if not show_mask:
                        cv2.destroyWindow(f"{args.title} - mask")
                reader.wait_new_frame(timeout_ms=max(1, int(args.poll_ms)))
                continue

            shm_start = time.perf_counter()
            hdr, frame_view = reader.read_latest_bgra()
            if hdr is None or frame_view is None:
                continue

            frame = numpy.frombuffer(frame_view, dtype=numpy.uint8).copy()
            frame = frame.reshape((hdr.height, hdr.pitch))
            frame = frame[:, : hdr.width * 4].reshape((hdr.height, hdr.width, 4))
            last_frame_id = hdr.frame_id
            shm_ms = (time.perf_counter() - shm_start) * 1000.0

            now_ms = int(time.time() * 1000)
            if args.max_fps > 0:
                min_interval = int(1000.0 / args.max_fps)
                if now_ms - last_show_ms < min_interval:
                    continue
            last_show_ms = now_ms

            cvt_start = time.perf_counter()
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            cvt_ms = (time.perf_counter() - cvt_start) * 1000.0

            wh = (int(hdr.width), int(hdr.height))
            if configured_size != wh:
                motion_saliency.setImagesize(wh[0], wh[1])
                motion_saliency.init()
                configured_size = wh

            flow_ms = 0.0
            global_motion_mask = None
            flow_info = None
            need_flow = bool(args.suppress_global_motion or args.show_flow or args.show_region_flow)
            if need_flow and prev_gray is not None:
                flow_start = time.perf_counter()
                flow_info = _compute_flow_info(
                    cv2,
                    numpy,
                    prev_gray,
                    gray,
                    flow_max_size=int(args.flow_max_size),
                )
                if args.suppress_global_motion:
                    global_motion_mask = _residual_motion_mask_from_flow(
                        cv2,
                        numpy,
                        flow_info,
                        full_w=int(hdr.width),
                        full_h=int(hdr.height),
                        residual_thr_px=float(args.residual_thr_px),
                        morph_ksize=int(args.morph_ksize),
                    )
                flow_ms = (time.perf_counter() - flow_start) * 1000.0
            prev_gray = gray

            sal_start = time.perf_counter()
            saliency = _compute_motion_saliency(motion_saliency, gray, numpy)
            if saliency is None:
                # Motion saliency needs a short history; skip drawing until ready.
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue
            saliency_ms = (time.perf_counter() - sal_start) * 1000.0

            draw_start = time.perf_counter()
            if global_motion_mask is not None:
                gated_saliency = saliency.astype(numpy.float32, copy=False) * (global_motion_mask.astype(numpy.float32) / 255.0)
                overlay, mask, boxes = _draw_saliency(cv2, numpy, bgr, gated_saliency, params)
            else:
                overlay, mask, boxes = _draw_saliency(cv2, numpy, bgr, saliency, params)

            if flow_info is not None and args.show_flow:
                _draw_flow_grid(
                    cv2,
                    numpy,
                    overlay,
                    flow_info,
                    step_px=int(args.flow_step),
                    arrow_scale=float(args.flow_arrow_scale),
                    min_mag=float(args.flow_min_mag),
                    use_residual=bool(args.flow_use_residual),
                    color=(0, 255, 255),
                )
            if flow_info is not None and args.show_region_flow and boxes:
                _draw_region_median_flow(
                    cv2,
                    numpy,
                    overlay,
                    flow_info,
                    boxes,
                    min_mag=float(args.flow_min_mag),
                    arrow_scale=float(args.flow_arrow_scale),
                    use_residual=bool(args.flow_use_residual),
                    color=(255, 0, 255),
                )

            shown_frames += 1
            elapsed_s = max(0.001, (now_ms - shown_start_ms) / 1000.0)
            fps = shown_frames / elapsed_s
            _put_hud_text(
                cv2,
                overlay,
                f"{hdr.width}x{hdr.height} frameId={hdr.frame_id} fps={fps:.1f} bbox={len(boxes)} thr={params.threshold:.2f}",
                margin=10,
                base_scale=0.6,
                thickness=1,
            )

            cv2.imshow(args.title, overlay)
            if show_mask:
                cv2.imshow(f"{args.title} - mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("m"),):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow(f"{args.title} - mask")

            draw_ms = (time.perf_counter() - draw_start) * 1000.0
            total_ms = (time.perf_counter() - loop_start) * 1000.0
            if args.print_perf:
                perf.add(
                    total_ms=total_ms,
                    shm_ms=shm_ms,
                    cvt_ms=cvt_ms,
                    flow_ms=flow_ms,
                    saliency_ms=saliency_ms,
                    draw_ms=draw_ms,
                )
                perf.maybe_print(now_ms=int(time.time() * 1000))
    finally:
        try:
            reader.close()
        except Exception as exc:
            print(f"[videoshm] close failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
