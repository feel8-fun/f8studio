#!/usr/bin/env python3
from __future__ import annotations

"""
UDP skeleton protocol test stub.

Sends the same binary packet format that `f8.udp_skeleton` decodes:
- aligned (null-terminated, padded to 4 bytes) strings
- little-endian timestamp (u64), boneCount (i32), then per-bone 7 float32s

Examples:
  python scripts/udp_skeleton_stub.py --host 127.0.0.1 --port 39540 --model-count 6
  python scripts/udp_skeleton_stub.py --models "Alice,Bob,Charlie" --drop-frame-prob 0.05
"""

import argparse
import math
import os
import random
import socket
import struct
import time
from dataclasses import dataclass
from typing import Iterable


def _pack_aligned_string(value: str) -> bytes:
    raw = value.encode("utf-8") + b"\x00"
    pad = (4 - (len(raw) & 0x03)) & 0x03
    return raw + (b"\x00" * pad)


def _clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)


@dataclass
class BoneSpec:
    name: str
    rest_pos: tuple[float, float, float]
    phase: float


def _default_bones(n: int, *, seed: int) -> list[BoneSpec]:
    rng = random.Random(seed)
    bones: list[BoneSpec] = []
    for i in range(int(n)):
        bones.append(
            BoneSpec(
                name=f"bone_{i}",
                rest_pos=(rng.uniform(-0.2, 0.2), rng.uniform(0.8, 1.8), rng.uniform(-0.2, 0.2)),
                phase=rng.uniform(0.0, math.tau),
            )
        )
    return bones


def _quat_from_yaw(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw_rad
    return (math.cos(half), 0.0, math.sin(half), 0.0)  # (qw, qx, qy, qz)


def build_skeleton_packet(
    *,
    model_name: str,
    timestamp_ms: int,
    schema: str,
    bones: Iterable[BoneSpec],
    t_s: float,
) -> bytes:
    out = bytearray()
    out += _pack_aligned_string(model_name)
    out += struct.pack("<Q", int(timestamp_ms))
    out += _pack_aligned_string(schema)

    bones_list = list(bones)
    out += struct.pack("<i", int(len(bones_list)))

    # Simple periodic motion: small bob + slow yaw.
    bob = 0.03 * math.sin(2.0 * math.pi * 1.0 * t_s)
    yaw = 0.35 * math.sin(2.0 * math.pi * 0.25 * t_s)
    qw, qx, qy, qz = _quat_from_yaw(yaw)

    for b in bones_list:
        out += _pack_aligned_string(b.name)
        px, py, pz = b.rest_pos
        wiggle = 0.015 * math.sin(2.0 * math.pi * 0.9 * t_s + b.phase)
        x = float(px + wiggle)
        y = float(py + bob)
        z = float(pz - wiggle)
        out += struct.pack("<fffffff", x, y, z, float(qw), float(qx), float(qy), float(qz))

    return bytes(out)


def _parse_model_names(raw: str, *, count: int) -> list[str]:
    raw = (raw or "").strip()
    if raw:
        parts = [p.strip() for p in raw.replace(";", ",").split(",")]
        names = [p for p in parts if p]
        if names:
            return names
    return [f"Model_{i+1}" for i in range(int(count))]


def main() -> int:
    ap = argparse.ArgumentParser(description="UDP skeleton protocol test stub (60fps sender).")
    ap.add_argument("--host", default=os.environ.get("F8_UDP_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("F8_UDP_PORT", "39540")))
    ap.add_argument("--fps", type=float, default=60.0, help="Send rate (frames per second).")
    ap.add_argument(
        "--models",
        default="",
        help="Comma-separated model names (default: Model_1..N).",
    )
    ap.add_argument("--model-count", type=int, default=4, help="How many models if --models is empty.")
    ap.add_argument("--bone-count", type=int, default=24)
    ap.add_argument("--schema", default="f8.skeleton.v1")
    ap.add_argument(
        "--drop-frame-prob",
        type=float,
        default=0.02,
        help="Probability to drop an entire frame (all models) to simulate packet loss.",
    )
    ap.add_argument(
        "--rename-after-s",
        type=float,
        default=5.0,
        help="After this many seconds, rename ~half the models (one-time).",
    )
    ap.add_argument("--duration-s", type=float, default=0.0, help="Stop after N seconds (0 = run forever).")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.port <= 0 or args.port >= 65536:
        raise SystemExit(f"Invalid --port: {args.port}")
    fps = float(args.fps)
    if fps <= 0:
        raise SystemExit(f"Invalid --fps: {args.fps}")
    drop_p = _clamp01(float(args.drop_frame_prob))

    rng = random.Random(int(args.seed))
    model_names = _parse_model_names(str(args.models), count=int(args.model_count))
    bones_by_model: dict[str, list[BoneSpec]] = {
        name: _default_bones(int(args.bone_count), seed=rng.randrange(1_000_000_000)) for name in model_names
    }

    addr = (str(args.host), int(args.port))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    start_wall = time.time()
    start_mono = time.monotonic()
    frame_period_s = 1.0 / fps
    next_deadline = time.monotonic()
    renamed = False

    if not args.quiet:
        print(f"Sending to udp://{addr[0]}:{addr[1]} at {fps:g} fps, models={len(model_names)}, bones={args.bone_count}")

    frames = 0
    dropped = 0
    sent_packets = 0
    last_report = time.monotonic()

    try:
        while True:
            now_mono = time.monotonic()
            if args.duration_s and (now_mono - start_mono) >= float(args.duration_s):
                break

            sleep_s = next_deadline - now_mono
            if sleep_s > 0:
                time.sleep(sleep_s)

            t_s = time.monotonic() - start_mono
            if (not renamed) and float(args.rename_after_s) > 0 and t_s >= float(args.rename_after_s):
                renamed = True
                n = len(model_names)
                k = max(1, n // 2) if n > 1 else 1
                indices = list(range(n))
                rng.shuffle(indices)
                chosen = set(indices[:k])
                old = list(model_names)
                model_names = [
                    (f"{name}_Renamed" if i in chosen else name)
                    for i, name in enumerate(old)
                ]
                # Preserve per-model bones for old names; copy to new names where needed.
                for i, (old_name, new_name) in enumerate(zip(old, model_names)):
                    if new_name != old_name:
                        bones_by_model[new_name] = bones_by_model.pop(old_name)
                if not args.quiet:
                    print(f"Renamed {k}/{n} models at t={t_s:.2f}s: {', '.join([m for m in model_names if m.endswith('_Renamed')])}")

            frames += 1
            if rng.random() < drop_p:
                dropped += 1
                next_deadline += frame_period_s
                continue

            ts_ms = int(time.time() * 1000)
            for name in list(model_names):
                pkt = build_skeleton_packet(
                    model_name=name,
                    timestamp_ms=ts_ms,
                    schema=str(args.schema),
                    bones=bones_by_model[name],
                    t_s=t_s,
                )
                sock.sendto(pkt, addr)
                sent_packets += 1

            next_deadline += frame_period_s

            if not args.quiet and (time.monotonic() - last_report) >= 2.0:
                last_report = time.monotonic()
                eff_fps = frames / max(1e-6, (time.monotonic() - start_mono))
                print(
                    f"frames={frames} dropped={dropped} sent_packets={sent_packets} eff_fps={eff_fps:.1f} models={len(model_names)}"
                )
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sock.close()
        except Exception:
            pass

    if not args.quiet:
        elapsed = max(1e-6, time.monotonic() - start_mono)
        print(
            f"Done: frames={frames} dropped={dropped} sent_packets={sent_packets} elapsed_s={elapsed:.2f} avg_fps={frames/elapsed:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
