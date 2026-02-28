from __future__ import annotations

from typing import Any

import msgpack


def encode_obj(obj: dict[str, Any]) -> bytes:
    """
    Encode a dictionary payload into MsgPack bytes.

    Raises:
        TypeError: if payload is not a dictionary.
        ValueError: if encoding fails.
    """
    if not isinstance(obj, dict):
        raise TypeError("payload must be a dict")
    try:
        return msgpack.packb(obj, use_bin_type=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"msgpack encode failed: {exc}") from exc


def decode_obj(raw: bytes) -> dict[str, Any]:
    """
    Decode MsgPack bytes into a dictionary payload.

    Raises:
        ValueError: if bytes cannot be decoded to a dictionary.
    """
    if not raw:
        return {}
    try:
        decoded = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    except (TypeError, ValueError, msgpack.ExtraData, msgpack.FormatError, msgpack.StackError) as exc:
        raise ValueError(f"msgpack decode failed: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("msgpack payload is not a dict")
    return decoded
