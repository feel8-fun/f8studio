#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import struct
import time
from typing import Any


def _read_aligned_string(buf: bytes, offset: int) -> tuple[str, int]:
    end = buf.find(b"\x00", offset)
    if end < 0:
        raise ValueError("missing string terminator")
    value = buf[offset:end].decode("utf-8")
    end += 1
    pad = (4 - (end & 0x03)) & 0x03
    return value, end + pad


def decode_skeleton_packet(data: bytes) -> dict[str, Any] | None:
    offset = 0
    try:
        model_name, offset = _read_aligned_string(data, offset)
        if offset + 8 > len(data):
            return None
        (timestamp_ms,) = struct.unpack_from("<Q", data, offset)
        offset += 8

        schema, offset = _read_aligned_string(data, offset)
        if offset + 4 > len(data):
            return None
        (bone_count,) = struct.unpack_from("<i", data, offset)
        offset += 4
        if bone_count < 0 or bone_count > 100000:
            return None

        bones: list[dict[str, Any]] = []
        for _ in range(int(bone_count)):
            name, offset = _read_aligned_string(data, offset)
            if offset + 28 > len(data):
                return None
            x, y, z, qw, qx, qy, qz = struct.unpack_from("<fffffff", data, offset)
            offset += 28
            bones.append(
                {
                    "name": name,
                    "pos": [x, y, z],
                    "rot": [qw, qx, qy, qz],
                }
            )

        trailer = None
        if offset + 30 <= len(data) and data[offset : offset + 4] == b"LMEX":
            # LMEX + extVersion(u16) + frameId(u64) + chunkIndex(i32) + chunkCount(i32)
            # + totalBoneCount(i32) + characterId(i32)
            ext_ver, frame_id, chunk_i, chunk_n, total_bones, character_id = struct.unpack_from(
                "<HQiiii", data, offset + 4
            )
            trailer = {
                "magic": "LMEX",
                "extVersion": ext_ver,
                "frameId": frame_id,
                "chunkIndex": chunk_i,
                "chunkCount": chunk_n,
                "totalBoneCount": total_bones,
                "characterId": character_id,
            }
            ext_offset = offset + 30
            if ext_offset + 12 <= len(data) and data[ext_offset : ext_offset + 4] == b"ANIM":
                normalized_time = struct.unpack_from("<f", data, ext_offset + 4)[0]
                layer_index = struct.unpack_from("<i", data, ext_offset + 8)[0]
                clip_name, next_offset = _read_aligned_string(data, ext_offset + 12)
                pose_key, _ = _read_aligned_string(data, next_offset)
                trailer["anim"] = {
                    "normalizedTime": normalized_time,
                    "layerIndex": layer_index,
                    "clipName": clip_name,
                    "poseKey": pose_key,
                }

        return {
            "type": "skeleton_binary",
            "modelName": model_name,
            "timestampMs": int(timestamp_ms),
            "schema": schema,
            "boneCount": int(bone_count),
            "bones": bones,
            "trailer": trailer,
        }
    except (UnicodeDecodeError, struct.error, ValueError):
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
