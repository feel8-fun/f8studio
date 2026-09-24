from __future__ import annotations

import json

import msgspec

from .models import StudioDocument
from .validation import validate_document


_DOCUMENT_DECODER = msgspec.json.Decoder(StudioDocument)
_JSON_ENCODER = msgspec.json.Encoder()


def encode_document(document: StudioDocument) -> bytes:
    return _JSON_ENCODER.encode(document)


def decode_document(payload: bytes | str) -> StudioDocument:
    try:
        document = _DOCUMENT_DECODER.decode(payload)
    except msgspec.DecodeError as exc:
        raise ValueError(f"invalid Studio document: {exc}") from exc
    validate_document(document)
    return document


def clone_document(document: StudioDocument) -> StudioDocument:
    return _DOCUMENT_DECODER.decode(_JSON_ENCODER.encode(document))


def canonical_json_bytes(value: object) -> bytes:
    builtins = msgspec.to_builtins(value, str_keys=True)
    return json.dumps(builtins, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
