from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class GpuMetrics:
    utilization_percent: float | None
    memory_used_mb: float | None


@dataclass(frozen=True)
class FrameMetric:
    frame_index: int
    ts_ms: float
    infer_ms: float
    pred_count: int
    gpu_utilization_percent: float | None
    gpu_memory_mb: float | None


def _query_gpu_metrics(gpu_index: int) -> GpuMetrics:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
        "-i",
        str(int(gpu_index)),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=2.0)
    except Exception:
        return GpuMetrics(utilization_percent=None, memory_used_mb=None)
    line = str(out).strip().splitlines()
    if not line:
        return GpuMetrics(utilization_percent=None, memory_used_mb=None)
    parts = [x.strip() for x in line[0].split(",")]
    if len(parts) < 2:
        return GpuMetrics(utilization_percent=None, memory_used_mb=None)
    try:
        util = float(parts[0])
    except ValueError:
        util = None
    try:
        mem = float(parts[1])
    except ValueError:
        mem = None
    return GpuMetrics(utilization_percent=util, memory_used_mb=mem)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    q_clamped = min(1.0, max(0.0, float(q)))
    idx = q_clamped * float(len(sorted_values) - 1)
    lo = int(idx)
    hi = min(len(sorted_values) - 1, lo + 1)
    frac = idx - float(lo)
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark ONNX DL baseline for f8.dl services.")
    parser.add_argument("--model-yaml", required=True, help="Path to model yaml (f8onnxModel/1).")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--ort-provider", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for classification models.")
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process all frames.")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index for nvidia-smi query.")
    parser.add_argument("--output-csv", default="", help="Optional CSV path for per-frame metrics.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    model_yaml = Path(str(args.model_yaml)).expanduser().resolve()
    video_path = Path(str(args.video)).expanduser().resolve()
    if not model_yaml.exists():
        raise SystemExit(f"Model yaml does not exist: {model_yaml}")
    if not video_path.exists():
        raise SystemExit(f"Video does not exist: {video_path}")

    import cv2  # type: ignore

    from f8pydl.model_config import load_model_spec
    from f8pydl.onnx_runtime import OnnxClassifierRuntime, OnnxYoloDetectorRuntime

    spec = load_model_spec(model_yaml)
    runtime_kind: Literal["detector", "classifier"]
    detector_runtime: OnnxYoloDetectorRuntime | None = None
    classifier_runtime: OnnxClassifierRuntime | None = None
    if spec.task == "yolo_cls":
        classifier_runtime = OnnxClassifierRuntime(spec, ort_provider=args.ort_provider)
        runtime_kind = "classifier"
    else:
        detector_runtime = OnnxYoloDetectorRuntime(spec, ort_provider=args.ort_provider)
        runtime_kind = "detector"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    frame_metrics: list[FrameMetric] = []
    warmup_frames = max(0, int(args.warmup_frames))
    max_frames = max(0, int(args.max_frames))
    processed = 0
    measured = 0
    t_start = time.perf_counter()
    t_last_gpu = 0.0
    latest_gpu = GpuMetrics(utilization_percent=None, memory_used_mb=None)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            processed += 1
            if max_frames > 0 and measured >= max_frames:
                break

            if processed <= warmup_frames:
                if runtime_kind == "classifier":
                    assert classifier_runtime is not None
                    classifier_runtime.infer(frame, top_k=max(1, int(args.top_k)))
                else:
                    assert detector_runtime is not None
                    detector_runtime.infer(frame)
                continue

            now = time.perf_counter()
            if (now - t_last_gpu) >= 1.0:
                latest_gpu = _query_gpu_metrics(int(args.gpu_index))
                t_last_gpu = now

            t0 = time.perf_counter()
            if runtime_kind == "classifier":
                assert classifier_runtime is not None
                topk, _meta = classifier_runtime.infer(frame, top_k=max(1, int(args.top_k)))
                pred_count = len(topk)
            else:
                assert detector_runtime is not None
                dets, _meta = detector_runtime.infer(frame)
                pred_count = len(dets)
            t1 = time.perf_counter()

            ts_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            frame_metrics.append(
                FrameMetric(
                    frame_index=measured,
                    ts_ms=ts_ms,
                    infer_ms=(t1 - t0) * 1000.0,
                    pred_count=int(pred_count),
                    gpu_utilization_percent=latest_gpu.utilization_percent,
                    gpu_memory_mb=latest_gpu.memory_used_mb,
                )
            )
            measured += 1
    finally:
        cap.release()

    elapsed_s = max(1e-9, time.perf_counter() - t_start)
    infer_values = sorted(float(x.infer_ms) for x in frame_metrics)
    fps = float(len(frame_metrics)) / elapsed_s if frame_metrics else 0.0
    p50 = _percentile(infer_values, 0.5)
    p95 = _percentile(infer_values, 0.95)
    gpu_utils = [float(x.gpu_utilization_percent) for x in frame_metrics if x.gpu_utilization_percent is not None]
    gpu_mems = [float(x.gpu_memory_mb) for x in frame_metrics if x.gpu_memory_mb is not None]

    summary = {
        "schemaVersion": "f8dlBenchmark/1",
        "model": {"id": spec.model_id, "task": spec.task, "path": str(spec.onnx_path)},
        "video": str(video_path),
        "ortProvider": str(args.ort_provider),
        "warmupFrames": int(warmup_frames),
        "measuredFrames": int(len(frame_metrics)),
        "elapsedSec": float(elapsed_s),
        "fps": float(fps),
        "inferMsP50": float(p50),
        "inferMsP95": float(p95),
        "gpuUtilizationPercentAvg": (sum(gpu_utils) / float(len(gpu_utils))) if gpu_utils else None,
        "gpuMemoryMbAvg": (sum(gpu_mems) / float(len(gpu_mems))) if gpu_mems else None,
    }

    if str(args.output_csv).strip():
        csv_path = Path(str(args.output_csv)).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_index", "ts_ms", "infer_ms", "pred_count", "gpu_utilization_percent", "gpu_memory_mb"])
            for m in frame_metrics:
                writer.writerow(
                    [
                        int(m.frame_index),
                        float(m.ts_ms),
                        float(m.infer_ms),
                        int(m.pred_count),
                        m.gpu_utilization_percent if m.gpu_utilization_percent is not None else "",
                        m.gpu_memory_mb if m.gpu_memory_mb is not None else "",
                    ]
                )
        summary_path = csv_path.with_suffix(csv_path.suffix + ".summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
