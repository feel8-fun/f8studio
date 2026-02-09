from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from f8pysdk import F8ServiceDescribe
from f8pystudio.service_catalog.discovery import find_service_dirs, load_service_entry


def _extract_last_json_obj(text: str) -> Any | None:
    """
    Best-effort: extract the last JSON value from noisy output (logs + JSON).

    Handles nested objects by using `raw_decode` instead of naive brace matching.
    """
    s = (text or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass

    decoder = json.JSONDecoder()
    idx = 0
    last: Any | None = None
    while idx < len(s):
        m = re.search(r"[\{\[]", s[idx:])
        if not m:
            break
        start = idx + m.start()
        try:
            obj, end = decoder.raw_decode(s[start:])
            last = obj
            idx = start + end
        except Exception:
            idx = start + 1
    return last


def _run_describe_subprocess(service_dir: Path, *, timeout_s: float) -> dict[str, Any]:
    entry = load_service_entry(service_dir)
    launch = entry.launch
    describe_args = list(entry.describeArgs or ["--describe"])

    cmd = [str(launch.command), *[str(a) for a in (launch.args or [])], *[str(a) for a in describe_args]]
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (launch.env or {}).items()})

    proc = subprocess.run(
        cmd,
        cwd=str(Path(launch.workdir or service_dir).resolve()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=max(0.1, timeout_s),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"describe failed (rc={proc.returncode}) for {service_dir}: {' '.join(cmd)}\n"
            f"stdout:\n{(proc.stdout or '').strip()}\n"
            f"stderr:\n{(proc.stderr or '').strip()}"
        )

    obj = _extract_last_json_obj(proc.stdout or "")
    if not isinstance(obj, dict):
        raise RuntimeError(
            f"describe produced no JSON object for {service_dir}: {' '.join(cmd)}\n"
            f"stdout:\n{(proc.stdout or '').strip()}\n"
            f"stderr:\n{(proc.stderr or '').strip()}"
        )

    payload = F8ServiceDescribe.model_validate(obj).model_dump(mode="json")
    return payload


def _write_describe_json(service_dir: Path, payload: dict[str, Any]) -> None:
    out_path = service_dir / "describe.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate per-service describe.json to speed up Studio discovery.")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=["services"],
        help="Service discovery roots (default: ./services).",
    )
    parser.add_argument("--service-class", action="append", default=[], help="Only update these serviceClass ids.")
    parser.add_argument("--timeout-s", type=float, default=10.0, help="Timeout per describe subprocess (seconds).")
    args = parser.parse_args(argv)

    roots = [Path(p).expanduser().resolve() for p in args.roots]
    only_service_classes = {str(s).strip() for s in args.service_class if str(s).strip()}

    service_dirs = find_service_dirs(roots)
    if not service_dirs:
        print(f"No services found under: {', '.join(str(r) for r in roots)}", file=sys.stderr)
        return 2

    updated = 0
    for service_dir in service_dirs:
        entry = load_service_entry(service_dir)
        service_class = str(entry.serviceClass or "").strip()
        if only_service_classes and service_class not in only_service_classes:
            continue

        payload = _run_describe_subprocess(service_dir, timeout_s=float(args.timeout_s))
        _write_describe_json(service_dir, payload)
        updated += 1
        print(f"updated {service_class or service_dir.name}: {service_dir / 'describe.json'}")

    if only_service_classes and updated == 0:
        print(
            f"No matching services for --service-class: {', '.join(sorted(only_service_classes))}",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

