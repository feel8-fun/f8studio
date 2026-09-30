from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
import tempfile
from pathlib import Path
import re
from typing import Any, Protocol, cast

import msgspec


def _run_python_codegen(*, protocol_path: Path, output_path: Path) -> None:
    from datamodel_code_generator import DataModelType, InputFileType, generate

    generate(
        protocol_path,
        input_file_type=InputFileType.OpenAPI,
        output=output_path,
        output_model_type=DataModelType.MsgspecStruct,
        use_annotated=True,
        use_title_as_name=True,
        strict_nullable=True,
        field_constraints=True,
        apply_default_values_for_required_fields=True,
        keyword_only=True,
    )


class GeneratedProtocol(Protocol):
    F8RuntimeGraph: type[msgspec.Struct]
    F8SetRungraphRequest: type[msgspec.Struct]
    F8SetRungraphArgs: type[msgspec.Struct]
    F8SetRungraphReply: type[msgspec.Struct]
    F8CommandInvokeRequest: type[msgspec.Struct]
    F8CommandInvokeReply: type[msgspec.Struct]


def _import_module(module_path: Path) -> GeneratedProtocol:
    module_name = f"f8_generated_msgspec_{module_path.stat().st_mtime_ns}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return cast(GeneratedProtocol, module)


def _smoke_test_generated(output_path: Path) -> None:
    generated = _import_module(output_path)
    runtime_graph_type = generated.F8RuntimeGraph
    rungraph_req_type = generated.F8SetRungraphRequest
    rungraph_reply_type = generated.F8SetRungraphReply
    cmd_req_type = generated.F8CommandInvokeRequest
    cmd_reply_type = generated.F8CommandInvokeReply

    payload = {
        "graphId": "g-smoke",
        "revision": "r1",
        "nodes": [
            {
                "nodeId": "op1",
                "serviceId": "svc1",
                "serviceClass": "svc.a",
                "operatorClass": "svc.a.op1",
                "stateFields": [
                    {
                        "name": "x",
                        "valueSchema": {"type": "number"},
                        "access": "rw",
                    }
                ],
                "stateValues": {"x": 1.0},
            }
        ],
        "edges": [],
    }
    model = msgspec.convert(payload, type=runtime_graph_type)
    _ = msgspec.to_builtins(model)

    rungraph_req_payload = {
        "reqId": "req-rungraph-smoke",
        "args": {"graph": payload},
        "meta": {"source": "smoke"},
    }
    rungraph_req = msgspec.convert(rungraph_req_payload, type=rungraph_req_type)
    _ = msgspec.to_builtins(rungraph_req)

    cmd_req_payload = {
        "reqId": "req-cmd-smoke",
        "call": "ping",
        "args": {"x": 1},
        "meta": {"actor": "smoke"},
    }
    cmd_req = msgspec.convert(cmd_req_payload, type=cmd_req_type)
    _ = msgspec.to_builtins(cmd_req)

    cmd_reply_payload = {
        "reqId": "req-cmd-smoke",
        "ok": True,
        "result": {"pong": True},
        "error": None,
    }
    cmd_reply = msgspec.convert(cmd_reply_payload, type=cmd_reply_type)
    _ = msgspec.to_builtins(cmd_reply)


def _generated_public_names(source: str) -> list[str]:
    tree = ast.parse(source)
    names: list[str] = ["UNSET"]
    seen: set[str] = set(names)
    for node in tree.body:
        name: str | None = None
        if isinstance(node, ast.ClassDef):
            name = node.name
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id

        if name is None or name.startswith("_") or name in seen:
            continue
        names.append(name)
        seen.add(name)
    return names


def _render_generated_all(names: list[str]) -> str:
    lines = ["", "__all__ = ["]
    lines.extend(f'    "{name}",' for name in names)
    lines.append("]")
    return "\n".join(lines) + "\n"


def _postprocess_generated(output_path: Path) -> None:
    source = output_path.read_text(encoding="utf-8")
    updated = re.sub(
        r"class F8ComponentRecord\(Struct, kw_only=True\):",
        "class F8ComponentRecord(Struct, kw_only=True, forbid_unknown_fields=True):",
        source,
        count=1,
    )
    if updated == source:
        raise RuntimeError("generated module is missing F8ComponentRecord for post-processing")
    public_names = _generated_public_names(updated)
    if "F8RuntimeGraph" not in public_names:
        raise RuntimeError("generated module is missing F8RuntimeGraph for __all__ post-processing")
    updated = updated.rstrip() + _render_generated_all(public_names)
    output_path.write_text(updated, encoding="utf-8")


def _generate(*, protocol_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_python_codegen(protocol_path=protocol_path, output_path=output_path)
    _postprocess_generated(output_path)
    _smoke_test_generated(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate msgspec models from OpenAPI protocol schema.")
    parser.add_argument("--protocol", required=True, help="Path to protocol OpenAPI YAML")
    parser.add_argument("--output", required=True, help="Path to generated python output file")
    parser.add_argument("--check", action="store_true", help="Verify generated models without writing them")
    args = parser.parse_args()

    protocol_path = Path(str(args.protocol)).resolve()
    output_path = Path(str(args.output)).resolve()
    if args.check:
        with tempfile.TemporaryDirectory(prefix="f8-protocol-check-") as directory:
            generated = Path(directory) / "models.py"
            _generate(protocol_path=protocol_path, output_path=generated)
            def without_timestamp(path: Path) -> str:
                return "\n".join(line for line in path.read_text(encoding="utf-8").splitlines()
                                 if not line.startswith("#   timestamp:"))
            if not output_path.exists() or without_timestamp(generated) != without_timestamp(output_path):
                raise SystemExit(f"Generated protocol models are stale: {output_path}")
    else:
        _generate(protocol_path=protocol_path, output_path=output_path)


if __name__ == "__main__":
    main()
