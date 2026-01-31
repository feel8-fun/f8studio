from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LetterboxResult:
    image_bgr: Any
    scale: float
    pad_x: int
    pad_y: int


def iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = a_area + b_area - inter
    return float(inter) / float(denom) if denom > 0 else 0.0


def clamp_xyxy(xyxy: tuple[int, int, int, int], *, size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(int(x1), int(w - 1)))
    y1 = max(0, min(int(y1), int(h - 1)))
    x2 = max(0, min(int(x2), int(w)))
    y2 = max(0, min(int(y2), int(h)))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return int(x1), int(y1), int(x2), int(y2)


def xyxy_to_xywh(xyxy: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    return int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    return x1, y1, x2, y2


def nms_xyxy(boxes: Any, scores: Any, *, iou_thr: float) -> list[int]:
    import numpy as np  # type: ignore

    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32, copy=False)
    scores = scores.astype(np.float32, copy=False)
    order = scores.argsort()[::-1]

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        denom = areas[i] + areas[rest] - inter
        iou = np.where(denom > 0, inter / denom, 0.0)
        order = rest[iou <= float(iou_thr)]

    return keep


def letterbox_bgr(
    img_bgr: Any,
    *,
    new_shape: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
) -> LetterboxResult:
    import cv2  # type: ignore

    h0, w0 = img_bgr.shape[:2]
    new_w, new_h = int(new_shape[0]), int(new_shape[1])
    r = min(new_w / float(w0), new_h / float(h0)) if w0 and h0 else 1.0
    if r <= 0:
        r = 1.0
    w1 = int(round(w0 * r))
    h1 = int(round(h0 * r))
    img = cv2.resize(img_bgr, (w1, h1), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - w1
    pad_h = new_h - h1
    left = int(pad_w // 2)
    right = int(pad_w - left)
    top = int(pad_h // 2)
    bottom = int(pad_h - top)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return LetterboxResult(image_bgr=img, scale=float(r), pad_x=left, pad_y=top)

