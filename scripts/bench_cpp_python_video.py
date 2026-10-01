#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_SDK = ROOT / "sdk" / "python"
if str(PKG_SDK) not in sys.path:
    sys.path.insert(0, str(PKG_SDK))

from f8pysdk.video_transport import (  # noqa: E402
    LatestVideoFrameTransport,
    ZenohLatestVideoFrameTransport,
)


_FRAME_TS_HEADER = struct.Struct("<q")


@dataclass(frozen=True)
class CrossLangVideoStats:
    name: str
    backend: str
    channels: int
    width: int
    height: int
    payload_bytes: int
    target_fps: float
    iterations: int
    warmup_iterations: int
    delivered: int
    lost: int
    producer_published: int
    producer_failed: int
    producer_elapsed_s: float
    consumer_elapsed_s: float
    producer_publish_fps: float
    consumer_delivered_fps: float
    latency_avg_ms: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    latency_p99_ms: float | None
    min_ms: float | None
    max_ms: float | None
    ok: bool
    note: str


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (float(pct) / 100.0) * float(len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - float(low)
    return float(sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac)


def _read_payload_timestamp_ns(frame_payload: memoryview) -> int | None:
    if len(frame_payload) < _FRAME_TS_HEADER.size:
        return None
    try:
        return int(_FRAME_TS_HEADER.unpack_from(frame_payload, 0)[0])
    except (BufferError, TypeError, ValueError, struct.error):
        return None


def _wait_for_ready_file(path: Path, proc: subprocess.Popen[str], timeout_s: float) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if path.exists():
            return
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=1.0)
            raise RuntimeError(
                f"C++ publisher exited before ready file was created rc={proc.returncode} "
                f"stdout={stdout!r} stderr={stderr!r}"
            )
        time.sleep(0.01)
    raise TimeoutError(f"C++ publisher did not create ready file within {timeout_s:g}s: {path}")


def _open_consumer(
    *,
    key_expr: str,
    shm_pool_bytes: int,
) -> LatestVideoFrameTransport:
    return ZenohLatestVideoFrameTransport.open_subscriber(
        key_expr,
        shm_pool_bytes=int(shm_pool_bytes),
    )


def _producer_command(
    *,
    producer_bin: Path,
    key_expr: str,
    ready_file: Path,
    width: int,
    height: int,
    channels: int,
    fps: float,
    iterations: int,
    warmup_iterations: int,
    start_delay_ms: int,
    shm_pool_bytes: int,
) -> list[str]:
    cmd = [
        "pixi",
        "run",
        "-e",
        "cpp",
        str(producer_bin),
        "--ready-file",
        str(ready_file),
        "--video-width",
        str(int(width)),
        "--video-height",
        str(int(height)),
        "--channels",
        str(int(channels)),
        "--fps",
        str(float(fps)),
        "--iterations",
        str(int(iterations)),
        "--warmup-iterations",
        str(int(warmup_iterations)),
        "--start-delay-ms",
        str(int(start_delay_ms)),
        "--zenoh-shm-pool-bytes",
        str(int(shm_pool_bytes)),
        "--key",
        key_expr,
    ]
    return cmd


def _collect_frames(
    *,
    consumer: LatestVideoFrameTransport,
    producer: subprocess.Popen[str],
    iterations: int,
    warmup_iterations: int,
    target_fps: float,
) -> tuple[list[int], set[int], float]:
    expected_elapsed = float(iterations + warmup_iterations) / max(1.0, float(target_fps))
    deadline = time.monotonic() + expected_elapsed + 5.0
    latencies_ns: list[int] = []
    delivered_frame_ids: set[int] = set()
    start = time.monotonic()
    while time.monotonic() < deadline:
        if len(delivered_frame_ids) >= int(iterations):
            break
        if producer.poll() is not None and time.monotonic() > deadline - 4.0:
            break
        frame = consumer.wait_latest(timeout_ms=100)
        if frame is None:
            continue
        try:
            frame_id = int(frame.frame_id)
            if frame_id <= int(warmup_iterations):
                continue
            if frame_id in delivered_frame_ids:
                continue
            sent_ns = _read_payload_timestamp_ns(frame.payload)
            if sent_ns is None:
                continue
            delivered_frame_ids.add(frame_id)
            latencies_ns.append(int(time.monotonic_ns()) - int(sent_ns))
        finally:
            frame.release()
    return latencies_ns, delivered_frame_ids, time.monotonic() - start


def _run_one(
    *,
    producer_bin: Path,
    channels: int,
    width: int,
    height: int,
    fps: float,
    iterations: int,
    warmup_iterations: int,
    start_delay_ms: int,
    shm_pool_bytes: int,
) -> CrossLangVideoStats:
    run_id = uuid.uuid4().hex
    backend = "zenoh"
    key_expr = f"f8/bench/cpp_python/video/{run_id}/zenoh/{channels}"
    payload_bytes = int(width) * int(height) * int(channels)

    with tempfile.TemporaryDirectory(prefix="f8-cpp-python-video-") as temp_raw:
        ready_file = Path(temp_raw) / "publisher.ready"
        cmd = _producer_command(
            producer_bin=producer_bin,
            key_expr=key_expr,
            ready_file=ready_file,
            width=width,
            height=height,
            channels=channels,
            fps=fps,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            start_delay_ms=start_delay_ms,
            shm_pool_bytes=shm_pool_bytes,
        )
        producer = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        consumer: LatestVideoFrameTransport | None = None
        try:
            _wait_for_ready_file(ready_file, producer, timeout_s=10.0)
            consumer = _open_consumer(
                key_expr=key_expr,
                shm_pool_bytes=shm_pool_bytes,
            )
            latencies_ns, delivered_frame_ids, consumer_elapsed_s = _collect_frames(
                consumer=consumer,
                producer=producer,
                iterations=iterations,
                warmup_iterations=warmup_iterations,
                target_fps=fps,
            )
            stdout, stderr = producer.communicate(timeout=10.0)
        finally:
            if consumer is not None:
                consumer.close()
            if producer.poll() is None:
                producer.terminate()
                try:
                    producer.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    producer.kill()
                    producer.wait(timeout=3.0)

    if producer.returncode != 0:
        raise RuntimeError(f"C++ publisher failed backend={backend} rc={producer.returncode} stderr={stderr!r}")
    try:
        producer_stats = json.loads(stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to parse C++ publisher output: {stdout!r}") from exc

    lat_ms = sorted(float(item) / 1_000_000.0 for item in latencies_ns)
    delivered = len(delivered_frame_ids)
    producer_published = int(producer_stats.get("published") or 0)
    producer_failed = int(producer_stats.get("failed") or 0)
    producer_elapsed_s = float(producer_stats.get("elapsed_s") or 0.0)
    measured_published = max(0, producer_published - int(warmup_iterations))
    producer_publish_fps = float(measured_published) / producer_elapsed_s if producer_elapsed_s > 0.0 else 0.0
    consumer_delivered_fps = float(delivered) / consumer_elapsed_s if consumer_elapsed_s > 0.0 else 0.0
    name = f"cpp_to_python.{backend}.video.{int(width)}x{int(height)}x{int(channels)}.{float(fps):g}fps"
    return CrossLangVideoStats(
        name=name,
        backend=backend,
        channels=int(channels),
        width=int(width),
        height=int(height),
        payload_bytes=payload_bytes,
        target_fps=float(fps),
        iterations=int(iterations),
        warmup_iterations=int(warmup_iterations),
        delivered=delivered,
        lost=max(0, int(iterations) - delivered),
        producer_published=producer_published,
        producer_failed=producer_failed,
        producer_elapsed_s=producer_elapsed_s,
        consumer_elapsed_s=consumer_elapsed_s,
        producer_publish_fps=producer_publish_fps,
        consumer_delivered_fps=consumer_delivered_fps,
        latency_avg_ms=(statistics.fmean(lat_ms) if lat_ms else None),
        latency_p50_ms=_percentile(lat_ms, 50.0),
        latency_p95_ms=_percentile(lat_ms, 95.0),
        latency_p99_ms=_percentile(lat_ms, 99.0),
        min_ms=(lat_ms[0] if lat_ms else None),
        max_ms=(lat_ms[-1] if lat_ms else None),
        ok=(delivered == int(iterations) and producer_failed == 0),
        note="C++ publisher, Python latest-frame consumer; latency uses payload monotonic timestamp",
    )


def _write_csv(path: Path, rows: list[CrossLangVideoStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(CrossLangVideoStats.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _print_csv(rows: list[CrossLangVideoStats]) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark C++ video producer -> Python latest-frame consumer.")
    parser.add_argument(
        "--producer-bin",
        type=Path,
        required=True,
        help="Path to the built f8cpp_crosslang_video_publisher executable (.exe on Windows)",
    )
    parser.add_argument("--video-width", type=int, default=1920)
    parser.add_argument("--video-height", type=int, default=1080)
    parser.add_argument("--channels", type=int, nargs="+", default=[3])
    parser.add_argument("--fps", type=float, nargs="+", default=[60.0, 120.0])
    parser.add_argument("--iterations", type=int, default=240)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--start-delay-ms", type=int, default=500)
    parser.add_argument("--zenoh-shm-pool-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    producer_bin = Path(args.producer_bin).resolve()
    if not producer_bin.is_file():
        raise FileNotFoundError(f"missing C++ producer binary: {producer_bin}")

    rows: list[CrossLangVideoStats] = []
    for channels in args.channels:
        if int(channels) not in (3, 4):
            raise ValueError("--channels entries must be 3 or 4")
        for fps in args.fps:
            rows.append(
                _run_one(
                    producer_bin=producer_bin,
                    channels=int(channels),
                    width=int(args.video_width),
                    height=int(args.video_height),
                    fps=float(fps),
                    iterations=int(args.iterations),
                    warmup_iterations=int(args.warmup_iterations),
                    start_delay_ms=int(args.start_delay_ms),
                    shm_pool_bytes=int(args.zenoh_shm_pool_bytes),
                )
            )

    _print_csv(rows)
    if args.output_csv is not None:
        _write_csv(Path(args.output_csv), rows)
    if args.output_json is not None:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
