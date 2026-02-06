#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _ExeCandidate:
    path: Path
    score: tuple[int, float]


def _repo_root_default() -> Path:
    # scripts/run_cpp_service.py -> repo root
    return Path(__file__).resolve().parent.parent


def _is_windows() -> bool:
    return os.name == "nt" or platform.system().lower().startswith("win")


def _normalize_path_str(s: str) -> str:
    s = str(s or "").strip().strip('"').strip("'")
    if not _is_windows():
        # Allow Windows-style paths in env/config (e.g. build\\bin\\foo.exe).
        s = s.replace("\\", "/")
    return s


def _truthy_env(name: str, *, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if s in ("", "0", "false", "no", "off", "disable", "disabled"):
        return False
    return True


def _prepare_isolated_conan_generators_dir(generators_dir: Path) -> Path | None:
    """
    Work around Conan PowerShell runenv scripts writing shared files under
    build/generators (eg. deactivate_conanrunenv-*.ps1). When multiple services
    start concurrently, those Out-File writes can race.

    Returns a temp directory that contains a copy of generators_dir, or None on
    failure/disabled.
    """
    if not _is_windows():
        return None
    if not generators_dir.is_dir():
        return None
    if not _truthy_env("F8_CONAN_RUNENV_ISOLATE", default=True):
        return None

    try:
        base = Path(tempfile.gettempdir()).resolve()
    except Exception:
        base = None  # type: ignore[assignment]
    if base is None:
        return None

    try:
        # Use a short, unique dir name to avoid long path issues on Windows.
        stamp = int(time.time() * 1000)
        out = Path(tempfile.mkdtemp(prefix=f"f8_conan_gen_{os.getpid()}_{stamp}_", dir=str(base)))
    except Exception:
        return None

    try:
        # Copy the whole generators folder to preserve relative-path assumptions in conanrun.ps1.
        # This folder is usually small (scripts + env files).
        shutil.copytree(str(generators_dir), str(out), dirs_exist_ok=True)
        return out
    except Exception:
        try:
            shutil.rmtree(str(out), ignore_errors=True)
        except Exception:
            pass
        return None


def _possible_filenames(exe: str) -> list[str]:
    exe = _normalize_path_str(exe)
    if not exe:
        return []

    # Allow users to pass a filename with extension (e.g. foo.exe).
    if Path(exe).suffix:
        return [exe]

    # Try both variants: Linux builds commonly have no extension; Windows builds do.
    return [exe, f"{exe}.exe"]


def _score_path(*, repo_root: Path, service_dir: Path | None, path: Path) -> tuple[int, float]:
    """
    Smaller score is better. Prefer:
      1) <repo>/build/bin/<exe>
      2) <service>/bin/<exe>
      3) anything else, newest first
    """
    try:
        p = path.resolve()
    except Exception:
        p = path

    s0 = 5

    try:
        rel = p.relative_to(repo_root.resolve())
        parts = {x.lower() for x in rel.parts}
        if "build" in parts and "bin" in parts:
            s0 = 0
    except Exception:
        pass

    if service_dir is not None:
        try:
            rel = p.relative_to(service_dir.resolve())
            parts = {x.lower() for x in rel.parts}
            if "bin" in parts:
                s0 = min(s0, 1)
        except Exception:
            pass

    try:
        mtime = p.stat().st_mtime
    except Exception:
        mtime = 0.0

    # Newer is better -> negate so "smaller is better" ranking still works.
    return (s0, -float(mtime))


def _collect_candidates(
    *,
    repo_root: Path,
    service_dir: Path | None,
    exe_names: list[str],
    build_dirs: list[Path],
) -> list[_ExeCandidate]:
    candidates: list[_ExeCandidate] = []

    def add_if_exists(p: Path) -> None:
        try:
            if p.is_file():
                candidates.append(_ExeCandidate(path=p, score=_score_path(repo_root=repo_root, service_dir=service_dir, path=p)))
        except Exception:
            return

    # Common (fast) locations.
    for name in exe_names:
        for bd in build_dirs:
            add_if_exists(bd / "bin" / name)
        if service_dir is not None:
            add_if_exists(service_dir / "bin" / name)

    # Fallback: search build tree (only if needed).
    if not candidates:
        for bd in build_dirs:
            if not bd.is_dir():
                continue
            for name in exe_names:
                try:
                    for p in bd.rglob(name):
                        add_if_exists(p)
                except Exception:
                    continue

    # Fallback: search service bin tree (only if needed). This supports packaged layouts like bin/linux-x64/<exe>.
    if not candidates and service_dir is not None:
        svc_bin = service_dir / "bin"
        if svc_bin.is_dir():
            for name in exe_names:
                try:
                    for p in svc_bin.rglob(name):
                        add_if_exists(p)
                except Exception:
                    continue

    return candidates


def _default_build_dirs(repo_root: Path) -> list[Path]:
    """
    Heuristics for common CMake build folder layouts used across platforms.
    """
    repo_root = Path(repo_root).resolve()

    env = _normalize_path_str(os.environ.get("F8_CPP_BUILD_DIRS") or "")
    if env:
        out: list[Path] = []
        for raw in [p for p in env.split(os.pathsep) if p.strip()]:
            p = Path(_normalize_path_str(raw)).expanduser()
            if not p.is_absolute():
                p = (repo_root / p).resolve()
            else:
                p = p.resolve()
            out.append(p)
        return out

    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path) -> None:
        try:
            p = p.resolve()
        except Exception:
            return
        if p in seen:
            return
        seen.add(p)
        candidates.append(p)

    add(repo_root / "build")
    add(repo_root / "out" / "build")

    # Common root-level patterns.
    for pat in ("build*", "cmake-build*", "_build*", ".build*", "out/build*"):
        try:
            for p in repo_root.glob(pat):
                if p.is_dir():
                    add(p)
        except Exception:
            continue

    # Also include immediate children of build/ when present (e.g. build/dev, build/linux-release).
    try:
        build_root = repo_root / "build"
        if build_root.is_dir():
            for child in build_root.iterdir():
                if child.is_dir():
                    add(child)
    except Exception:
        pass

    # Include immediate children of out/build/ when present (e.g. out/build/debug).
    try:
        out_build = repo_root / "out" / "build"
        if out_build.is_dir():
            for child in out_build.iterdir():
                if child.is_dir():
                    add(child)
    except Exception:
        pass

    return candidates


def _pick_executable(
    *,
    repo_root: Path,
    service_dir: Path | None,
    exe: str,
    env_var: str | None,
    build_dir: Path,
) -> Path:
    explicit = None
    if env_var:
        raw = os.environ.get(env_var)
        if raw:
            explicit = Path(_normalize_path_str(raw))

    if explicit is not None:
        if not explicit.is_absolute():
            explicit = (repo_root / explicit).resolve()
        if not explicit.exists():
            raise FileNotFoundError(f"{env_var} points to missing file: {explicit}")
        return explicit

    exe = _normalize_path_str(exe)
    # Allow passing an explicit relative/absolute executable path.
    if "/" in exe or ("\\" in exe and _is_windows()):
        p = Path(exe).expanduser()
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        else:
            p = p.resolve()
        if p.exists():
            return p
        # If the explicit path didn't exist, fall back to searching by filename.
        exe = p.name

    exe_names = _possible_filenames(exe)
    if not exe_names:
        raise ValueError("--exe is required")

    build_dirs: list[Path] = []
    try:
        build_dirs.append(build_dir.resolve())
    except Exception:
        pass
    for p in _default_build_dirs(repo_root):
        if p not in build_dirs:
            build_dirs.append(p)

    # Only walk directories that actually exist to keep search fast.
    existing_build_dirs = [p for p in build_dirs if p.is_dir()]
    if not existing_build_dirs:
        existing_build_dirs = build_dirs[:1]  # keep at least the hint for error messages

    cands = _collect_candidates(repo_root=repo_root, service_dir=service_dir, exe_names=exe_names, build_dirs=existing_build_dirs)
    if not cands:
        searched = [
            *[str(p / "bin") for p in existing_build_dirs],
            str((service_dir / "bin") if service_dir is not None else "<service-dir>/bin"),
            *[str(p) for p in existing_build_dirs],
        ]
        raise FileNotFoundError(
            f"{exe} not found.\n"
            f"Searched (in order):\n- " + "\n- ".join(searched)
        )

    cands.sort(key=lambda c: c.score)
    return cands[0].path.resolve()


def _exec_with_optional_conan_runenv(*, exe_path: Path, passthrough_args: list[str], auto_runenv: bool) -> None:
    # When running from a Conan build folder, prefer sourcing runenv scripts so deps are found.
    exe_dir = exe_path.parent
    build_root = exe_dir.parent if exe_dir.name.lower() == "bin" else exe_dir
    generators = build_root / "generators"

    if auto_runenv and generators.is_dir():
        if _is_windows():
            runenv_root = _prepare_isolated_conan_generators_dir(generators) or generators
            runenv = runenv_root / "conanrun.ps1"
            if runenv.is_file():
                def _ps_quote(s: str) -> str:
                    # PowerShell single-quoted strings escape quotes by doubling them.
                    return "'" + s.replace("'", "''") + "'"

                # Replace current process with PowerShell -> source runenv -> exec exe.
                cleanup = ""
                try:
                    if runenv_root != generators:
                        cleanup_dir = str(runenv_root)
                        cleanup = (
                            f"; $code=$LASTEXITCODE; "
                            f"try {{ Remove-Item -LiteralPath {_ps_quote(cleanup_dir)} -Recurse -Force -ErrorAction SilentlyContinue }} catch {{}}; "
                            f"exit $code"
                        )
                except Exception:
                    cleanup = ""
                cmd = [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    f". {_ps_quote(str(runenv))}; & {_ps_quote(str(exe_path))} "
                    + " ".join(_ps_quote(a) for a in passthrough_args)
                    + cleanup,
                ]
                os.execvp(cmd[0], cmd)
        else:
            runenv = generators / "conanrun.sh"
            if runenv.is_file():
                cmd_str = (
                    f"source {shlex.quote(str(runenv))} >/dev/null 2>&1 || true; "
                    f"exec {shlex.quote(str(exe_path))} " + " ".join(shlex.quote(a) for a in passthrough_args)
                )
                os.execvp("bash", ["bash", "-lc", cmd_str])

    os.execv(str(exe_path), [str(exe_path), *passthrough_args])


def _extract_last_json_obj(text: str) -> object | None:
    """
    Best-effort: extract the last JSON value from noisy output (logs + JSON).
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
    last: object | None = None
    while idx < len(s):
        pos_obj = s.find("{", idx)
        pos_arr = s.find("[", idx)
        if pos_obj == -1 and pos_arr == -1:
            break
        if pos_obj == -1:
            start = pos_arr
        elif pos_arr == -1:
            start = pos_obj
        else:
            start = min(pos_obj, pos_arr)
        try:
            obj, end = decoder.raw_decode(s[start:])
            last = obj
            idx = start + end
        except Exception:
            idx = start + 1
    return last


def _fallback_describe_payload(*, service_dir: Path | None) -> dict:
    payload: dict = {"service": {}, "operators": []}

    service_class = "unknown.service"
    label = "unknown.service"
    version = "0.0.0"

    if service_dir is not None:
        svc_yml = service_dir / "service.yml"
        try:
            import yaml  # type: ignore[import-not-found]

            raw = svc_yml.read_text("utf-8")
            parsed = yaml.safe_load(raw) if raw.strip() else None
            if isinstance(parsed, dict):
                service_class = str((parsed.get("serviceClass") or service_class) or "").strip() or service_class
                label = str((parsed.get("label") or label) or "").strip() or label
                version = str((parsed.get("version") or version) or "").strip() or version
        except Exception:
            pass

    if not re.match(r"^[0-9]+\.[0-9]+\.[0-9]+$", version):
        version = "0.0.0"

    payload["service"] = {"schemaVersion": "f8service/1", "serviceClass": service_class, "label": label, "version": version}
    return payload


def _normalize_describe_payload(*, payload: dict, service_dir: Path | None) -> dict:
    """
    Ensure payload is compatible with Studio registries:
    - service.schemaVersion must be f8service/1
    - operator.schemaVersion must be f8operator/1
    - ensure required labels exist (fallback to ids)
    """
    meta = _fallback_describe_payload(service_dir=service_dir)

    svc = payload.get("service")
    if not isinstance(svc, dict):
        payload["service"] = meta["service"]
    else:
        svc = dict(svc)
        svc.setdefault("schemaVersion", "f8service/1")
        svc.setdefault("serviceClass", meta["service"]["serviceClass"])
        svc.setdefault("label", meta["service"]["label"])
        svc.setdefault("version", meta["service"]["version"])
        if not re.match(r"^[0-9]+\.[0-9]+\.[0-9]+$", str(svc.get("version") or "")):
            svc["version"] = meta["service"]["version"]
        payload["service"] = svc

    ops = payload.get("operators")
    if not isinstance(ops, list):
        payload["operators"] = []
    else:
        fixed_ops: list[dict] = []
        for op in ops:
            if not isinstance(op, dict):
                continue
            o = dict(op)
            o.setdefault("schemaVersion", "f8operator/1")
            o.setdefault("serviceClass", payload["service"]["serviceClass"])
            o.setdefault("operatorClass", o.get("operatorClass") or "unknown.operator")
            o.setdefault("label", o.get("operatorClass"))
            if not re.match(r"^[0-9]+\.[0-9]+\.[0-9]+$", str(o.get("version") or "")):
                o["version"] = "0.0.0"
            fixed_ops.append(o)
        payload["operators"] = fixed_ops

    def _schema_string() -> dict:
        return {"type": "string"}

    def _state_field(*, name: str, value_schema: dict, access: str, label: str, description: str, show_on_node: bool) -> dict:
        return {
            "name": str(name),
            "label": str(label),
            "description": str(description),
            "valueSchema": dict(value_schema),
            "access": str(access),
            "showOnNode": bool(show_on_node),
        }

    def _ensure_state_field(obj: dict, *, name: str, access: str, label: str, description: str, show_on_node: bool) -> None:
        fields = obj.get("stateFields")
        if not isinstance(fields, list):
            fields = []
        have = {str(x.get("name") or "") for x in fields if isinstance(x, dict)}
        if name in have:
            obj["stateFields"] = fields
            return
        fields.append(
            _state_field(
                name=name,
                value_schema=_schema_string(),
                access=access,
                label=label,
                description=description,
                show_on_node=show_on_node,
            )
        )
        obj["stateFields"] = fields

    # Built-in identity fields (match f8pysdk behavior):
    # - svcId: for all nodes
    # - operatorId: operator nodes only
    try:
        _ensure_state_field(
            payload["service"],
            name="svcId",
            access="ro",
            label="Service Id",
            description="Readonly: current service instance id (svcId).",
            show_on_node=False,
        )
    except Exception:
        pass
    try:
        for op in list(payload.get("operators") or []):
            if not isinstance(op, dict):
                continue
            _ensure_state_field(
                op,
                name="svcId",
                access="ro",
                label="Service Id",
                description="Readonly: current service instance id (svcId).",
                show_on_node=False,
            )
            _ensure_state_field(
                op,
                name="operatorId",
                access="ro",
                label="Operator Id",
                description="Readonly: current operator/node id (operatorId).",
                show_on_node=False,
            )
    except Exception:
        pass

    return payload


def _run_describe(
    *,
    repo_root: Path,
    service_dir: Path | None,
    exe_path: Path | None,
    auto_runenv: bool,
    extra_args: list[str],
) -> dict:
    """
    Return describe payload dict. Never raises; falls back to minimal YAML-derived payload.
    """
    if exe_path is None:
        return _fallback_describe_payload(service_dir=service_dir)

    exe_dir = exe_path.parent
    build_root = exe_dir.parent if exe_dir.name.lower() == "bin" else exe_dir
    generators = build_root / "generators"

    try:
        if auto_runenv and generators.is_dir():
            if _is_windows() and (generators / "conanrun.ps1").is_file():
                def _ps_quote(s: str) -> str:
                    return "'" + s.replace("'", "''") + "'"

                runenv_root = _prepare_isolated_conan_generators_dir(generators) or generators
                runenv = runenv_root / "conanrun.ps1"
                cleanup = ""
                try:
                    if runenv_root != generators:
                        cleanup_dir = str(runenv_root)
                        cleanup = (
                            f"; $code=$LASTEXITCODE; "
                            f"try {{ Remove-Item -LiteralPath {_ps_quote(cleanup_dir)} -Recurse -Force -ErrorAction SilentlyContinue }} catch {{}}; "
                            f"exit $code"
                        )
                except Exception:
                    cleanup = ""
                cmd = [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    f". {_ps_quote(str(runenv))}; & {_ps_quote(str(exe_path))} --describe "
                    + " ".join(_ps_quote(a) for a in extra_args)
                    + cleanup,
                ]
                proc = subprocess.run(cmd, cwd=str(repo_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            elif not _is_windows() and (generators / "conanrun.sh").is_file():
                runenv = generators / "conanrun.sh"
                cmd_str = (
                    f"source {shlex.quote(str(runenv))} >/dev/null 2>&1 || true; "
                    f"{shlex.quote(str(exe_path))} --describe " + " ".join(shlex.quote(a) for a in extra_args)
                )
                proc = subprocess.run(["bash", "-lc", cmd_str], cwd=str(repo_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            else:
                proc = subprocess.run(
                    [str(exe_path), "--describe", *extra_args],
                    cwd=str(repo_root),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
        else:
            proc = subprocess.run(
                [str(exe_path), "--describe", *extra_args],
                cwd=str(repo_root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
    except Exception:
        return _fallback_describe_payload(service_dir=service_dir)

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    obj = _extract_last_json_obj(combined)
    if isinstance(obj, dict):
        # Preferred: already wrapped as f8describe/1.
        if "service" in obj:
            if "operators" not in obj:
                obj["operators"] = []
            obj.setdefault("schemaVersion", "f8describe/1")
            return _normalize_describe_payload(payload=obj, service_dir=service_dir)

        # Common C++ pattern: `--describe` prints only the f8service/1 service spec.
        # Wrap it so Studio discovery can consume it uniformly.
        if str(obj.get("schemaVersion") or "").strip() == "f8service/1" or "serviceClass" in obj:
            wrapped = {"schemaVersion": "f8describe/1", "service": obj, "operators": []}
            return _normalize_describe_payload(payload=wrapped, service_dir=service_dir)

    return _fallback_describe_payload(service_dir=service_dir)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="run_cpp_service.py",
        description="Cross-platform launcher for C++ services referenced by f8 service.yml (resolves build/bin vs service/bin).",
        add_help=True,
    )
    ap.add_argument("--repo-root", default=None, help="Repository root (default: inferred from this script).")
    ap.add_argument("--service-dir", default=None, help="Service directory containing service.yml (e.g. services/f8/implayer).")
    ap.add_argument("--exe", required=True, help="Executable base name (with or without .exe).")
    ap.add_argument("--env-var", default=None, help="Optional env var that overrides the executable path (relative paths are repo-root-relative).")
    ap.add_argument("--build-dir", default=None, help="Build directory to search (default: <repo-root>/build).")
    ap.add_argument("--print-only", action="store_true", help="Print resolved executable path and exit.")
    ap.add_argument("--describe", action="store_true", help="Output describe JSON (used by Studio service discovery).")
    ap.add_argument(
        "--no-conan-runenv",
        action="store_true",
        help="Do not attempt to source Conan runenv scripts (generators/conanrun.*).",
    )

    args, passthrough = ap.parse_known_args(argv)

    repo_root = Path(_normalize_path_str(args.repo_root)).expanduser().resolve() if args.repo_root else _repo_root_default()
    if args.service_dir:
        sd = Path(_normalize_path_str(args.service_dir)).expanduser()
        service_dir = (sd if sd.is_absolute() else (repo_root / sd)).resolve()
    else:
        service_dir = None
    build_dir = (
        Path(_normalize_path_str(args.build_dir)).expanduser().resolve()
        if args.build_dir
        else (repo_root / "build").resolve()
    )

    if args.describe:
        exe_path: Path | None
        try:
            exe_path = _pick_executable(
                repo_root=repo_root,
                service_dir=service_dir,
                exe=str(args.exe),
                env_var=(str(args.env_var) if args.env_var else None),
                build_dir=build_dir,
            )
        except Exception:
            exe_path = None

        payload = _run_describe(
            repo_root=repo_root,
            service_dir=service_dir,
            exe_path=exe_path,
            auto_runenv=not bool(args.no_conan_runenv),
            extra_args=passthrough,
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    exe_path = _pick_executable(
        repo_root=repo_root,
        service_dir=service_dir,
        exe=str(args.exe),
        env_var=(str(args.env_var) if args.env_var else None),
        build_dir=build_dir,
    )

    if args.print_only:
        print(str(exe_path))
        return 0

    _exec_with_optional_conan_runenv(
        exe_path=exe_path,
        passthrough_args=passthrough,
        auto_runenv=not bool(args.no_conan_runenv),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
