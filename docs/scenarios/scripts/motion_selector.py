# Motion-first selector for f8.python_script (Video SHM flow edition)
#
# Inputs:
#   - detections: f8visionDetections/1
#
# State:
#   - flowShm: dense flow SHM name (format flow2_f16)
#
# Output:
#   - selected: f8visionDetections/1 (single best detection)

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


MAX_FRAME_GAP = 2
MIN_FLOW_PIXELS_IN_ROI = 64
MIN_SCORE = 0.05
EMIT_MIN_INTERVAL_MS = 120


def _to_int(value, default=0):
    if isinstance(value, bool):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_float(value, default=0.0):
    if isinstance(value, bool):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bbox_xyxy(det, width, height):
    if not isinstance(det, dict):
        return None
    bbox = det.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    x1 = max(0, min(width, _to_int(bbox[0], 0)))
    y1 = max(0, min(height, _to_int(bbox[1], 0)))
    x2 = max(0, min(width, _to_int(bbox[2], 0)))
    y2 = max(0, min(height, _to_int(bbox[3], 0)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _frame_gap_ok(detections_payload, flow_header):
    det_frame = _to_int(detections_payload.get("frameId"), -1)
    flow_frame = _to_int(flow_header.get("frameId"), -1)
    if det_frame >= 0 and flow_frame >= 0:
        return abs(det_frame - flow_frame) <= MAX_FRAME_GAP
    det_ts = _to_int(detections_payload.get("tsMs"), -1)
    flow_ts = _to_int(flow_header.get("tsMs"), -1)
    if det_ts >= 0 and flow_ts >= 0:
        return abs(det_ts - flow_ts) <= 100
    return False


def _flow_array_from_packet(packet):
    if not isinstance(packet, dict):
        return None
    decoded = packet.get("decoded")
    if not isinstance(decoded, dict):
        return None
    if str(decoded.get("kind") or "") != "flow2_f16":
        return None
    data = decoded.get("data")
    if data is None:
        return None
    if np is None:
        return None
    if not isinstance(data, np.ndarray):
        return None
    if data.ndim != 3 or data.shape[2] != 2:
        return None
    return data.astype(np.float32, copy=False)


def _build_single_detection_payload(src_detections, picked_det, score):
    out = {
        "schemaVersion": str(src_detections.get("schemaVersion") or "f8visionDetections/1"),
        "frameId": _to_int(src_detections.get("frameId"), 0),
        "tsMs": _to_int(src_detections.get("tsMs"), 0),
        "width": _to_int(src_detections.get("width"), 0),
        "height": _to_int(src_detections.get("height"), 0),
        "model": src_detections.get("model"),
        "task": src_detections.get("task"),
        "skeletonProtocol": src_detections.get("skeletonProtocol"),
        "detections": [],
    }
    det_out = dict(picked_det)
    det_out["score"] = float(score)
    out["detections"] = [det_out]
    return out


def _select(detections_payload, flow_packet):
    if not isinstance(detections_payload, dict) or not isinstance(flow_packet, dict):
        return None
    flow_header = flow_packet.get("header")
    if not isinstance(flow_header, dict):
        return None
    if not _frame_gap_ok(detections_payload, flow_header):
        return None

    flow = _flow_array_from_packet(flow_packet)
    if flow is None:
        return None

    mag = np.sqrt(flow[:, :, 0] * flow[:, :, 0] + flow[:, :, 1] * flow[:, :, 1])
    global_mean = float(mag.mean()) if mag.size > 0 else 0.0
    height = int(flow.shape[0])
    width = int(flow.shape[1])

    detections = detections_payload.get("detections")
    if not isinstance(detections, list) or not detections:
        return None

    best_det = None
    best_score = -1e9
    for det in detections:
        bbox = _bbox_xyxy(det, width, height)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        roi = mag[y1:y2, x1:x2]
        roi_count = int(roi.size)
        if roi_count < MIN_FLOW_PIXELS_IN_ROI:
            continue
        roi_mean = float(roi.mean())
        motion_score = roi_mean - global_mean
        if motion_score > best_score:
            best_score = motion_score
            best_det = det

    if best_det is None or best_score < MIN_SCORE:
        return None
    return _build_single_detection_payload(detections_payload, best_det, best_score)


def _update_flow_subscription(ctx, flow_shm_name):
    if not isinstance(flow_shm_name, str):
        return
    name = flow_shm_name.strip()
    if not name:
        return
    ctx["locals"]["flowShm"] = name
    ctx["subscribe_video_shm"]("flow", name, decode="auto", use_event=False)


async def onStart(ctx):
    ctx["locals"]["latest_detections"] = None
    ctx["locals"]["last_emit_ts_ms"] = 0
    flow_shm = await ctx["get_state"]("flowShm")
    _update_flow_subscription(ctx, flow_shm)


def onState(ctx, field, value, tsMs=None):
    if str(field or "") == "flowShm":
        _update_flow_subscription(ctx, value)


def _handle(ctx, inputs):
    detections_payload = inputs.get("detections")
    if isinstance(detections_payload, dict):
        ctx["locals"]["latest_detections"] = detections_payload

    latest_detections = ctx["locals"].get("latest_detections")
    flow_packet = ctx["get_video_shm"]("flow")
    selected = _select(latest_detections, flow_packet)
    if selected is None:
        return None

    now_ts = _to_int(selected.get("tsMs"), 0)
    last_emit = _to_int(ctx["locals"].get("last_emit_ts_ms"), 0)
    if now_ts > 0 and last_emit > 0 and (now_ts - last_emit) < EMIT_MIN_INTERVAL_MS:
        return None

    ctx["locals"]["last_emit_ts_ms"] = now_ts
    return {"outputs": {"selected": selected}}


def onMsg(ctx, inputs):
    return _handle(ctx, inputs)


def onExec(ctx, execIn, inputs):
    return _handle(ctx, inputs)
