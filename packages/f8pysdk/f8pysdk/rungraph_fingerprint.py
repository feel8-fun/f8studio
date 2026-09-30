from __future__ import annotations

from typing import Any

from .codec import dump_json
from .generated.runtime_fingerprint import fingerprint, normalize_snapshot


def build_rungraph_deploy_snapshot(graph: Any) -> dict[str, Any]:
    return normalize_snapshot(dump_json(graph, mode="json", by_alias=True))


def build_rungraph_deploy_fingerprint(graph: Any) -> str:
    return fingerprint(dump_json(graph, mode="json", by_alias=True))


__all__ = ["build_rungraph_deploy_fingerprint", "build_rungraph_deploy_snapshot"]
