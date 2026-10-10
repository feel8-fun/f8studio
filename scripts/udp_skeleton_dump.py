#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Any

from f8pysdk.motion.skeleton_codec import SkeletonPacketDecodeError, decode_skeleton_packet as decode_sdk_skeleton_packet


def decode_skeleton_packet(data: bytes) -> dict[str, Any] | None:
    # A format probe: malformed/non-skeleton packets continue to text/raw output.
    try:
        return decode_sdk_skeleton_packet(data).to_payload()
    except SkeletonPacketDecodeError:
        return None


def decode_any(data: bytes) -> dict[str, Any]:
    decoded = decode_skeleton_packet(data)
    if decoded is not None:
        return decoded

    try:
        text = data.decode("utf-8")
        if not text:
            return {"type": "empty_text", "rawLen": len(data)}
        try:
            return {"type": "json_text", "payload": json.loads(text)}
        except json.JSONDecodeError:
            return {"type": "plain_text", "payload": text}
    except UnicodeDecodeError:
        return {
            "type": "raw_bytes",
            "rawLen": len(data),
            "hexPreview": data[:64].hex(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple UDP receiver + printer")
    parser.add_argument("--host", default="0.0.0.0", help="bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=39540, help="bind port (default: 39540)")
    parser.add_argument(
        "--max-bones",
        type=int,
        default=5,
        help="max number of bones to print in detail for skeleton packets (default: 5)",
    )
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(0.5)
    print(f"[udp_dump] listening on {args.host}:{args.port}")

    try:
        while True:
            try:
                data, addr = sock.recvfrom(1024 * 1024)
            except socket.timeout:
                continue
            now = int(time.time() * 1000)
            msg = decode_any(data)
            print(f"\n[{now}] from {addr[0]}:{addr[1]} len={len(data)} type={msg.get('type')}")

            if msg.get("type") == "skeleton_binary":
                print(
                    f"  model={msg['modelName']} schema={msg['schema']} "
                    f"timestampMs={msg['timestampMs']} bones={msg['boneCount']}"
                )
                bones = msg["bones"][: max(0, args.max_bones)]
                for i, b in enumerate(bones):
                    print(
                        f"    [{i}] {b['name']} "
                        f"pos={tuple(round(v, 4) for v in b['pos'])} "
                        f"rot={tuple(round(v, 4) for v in b['rot'])}"
                    )
                if msg["boneCount"] > len(bones):
                    print(f"    ... ({msg['boneCount'] - len(bones)} more bones)")
                if msg.get("trailer"):
                    print(f"  trailer={msg['trailer']}")
            else:
                print("  payload=" + json.dumps(msg.get("payload", msg), ensure_ascii=False))
    except KeyboardInterrupt:
        print("\n[udp_dump] stopped")
        return 0
    finally:
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
