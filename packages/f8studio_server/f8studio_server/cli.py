from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import cast
from uuid import uuid4

from .api_client import StudioApiClient, StudioApiError


def _object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _load_object(path: str) -> dict[str, object]:
    try:
        payload: object = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON file {path}: {exc}") from exc
    return _object(payload, label=path)


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _run_agent(client: StudioApiClient, args: argparse.Namespace) -> object:
    created = _object(
        client.create_agent_session(
            project_id=args.project_id,
            title=args.title,
            provider_id=args.provider,
            model_id=args.model,
        ),
        label="agent session",
    )
    session_id = str(created["sessionId"])
    client.start_agent_run(session_id, prompt=args.prompt)
    deadline = time.monotonic() + float(args.timeout)
    resolved: set[str] = set()
    while time.monotonic() < deadline:
        session = _object(client.agent_session(session_id), label="agent session")
        status = str(session.get("status") or "")
        if status in {"succeeded", "failed", "cancelled"}:
            return session
        approval_raw = session.get("approval")
        if isinstance(approval_raw, dict):
            approval = cast(dict[str, object], approval_raw)
        else:
            approval = {}
        if approval.get("status") == "pending":
            approval_id = str(approval.get("approvalId") or "")
            arguments_hash = str(approval.get("argumentsHash") or "")
            if args.approve and approval_id and approval_id not in resolved:
                client.resolve_agent_approval(
                    session_id,
                    approval_id,
                    approved=True,
                    arguments_hash=arguments_hash,
                )
                resolved.add(approval_id)
            elif not args.approve:
                return session
        time.sleep(0.05)
    raise TimeoutError(f"agent run did not finish within {args.timeout:.1f}s")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Feel8 Web Studio headless CLI")
    parser.add_argument("--url", default="http://127.0.0.1:8260", help="Studio Server base URL")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health")
    subparsers.add_parser("projects")
    subparsers.add_parser("catalog")
    graph = subparsers.add_parser("graph")
    graph.add_argument("project_id")
    preview = subparsers.add_parser("preview-patch")
    preview.add_argument("project_id")
    preview.add_argument("path")
    apply = subparsers.add_parser("apply-patch")
    apply.add_argument("project_id")
    apply.add_argument("path")
    validate = subparsers.add_parser("validate")
    validate.add_argument("project_id")
    deploy = subparsers.add_parser("deploy")
    deploy.add_argument("project_id")
    monitors = subparsers.add_parser("monitors")
    monitors.add_argument("project_id", nargs="?")
    providers = subparsers.add_parser("agent-providers")
    del providers
    agent = subparsers.add_parser("agent-run")
    agent.add_argument("project_id")
    agent.add_argument("prompt")
    agent.add_argument("--title", default="CLI agent session")
    agent.add_argument("--provider", default="deterministic")
    agent.add_argument("--model", default="graph-builder-v1")
    agent.add_argument("--approve", action="store_true", help="Approve exact pending tool calls")
    agent.add_argument("--timeout", type=float, default=30.0)
    agent_session = subparsers.add_parser("agent-session")
    agent_session.add_argument("session_id")
    agent_cancel = subparsers.add_parser("agent-cancel")
    agent_cancel.add_argument("session_id")
    return parser


def _dispatch(client: StudioApiClient, args: argparse.Namespace) -> object:
    if args.command == "health":
        return client.health()
    if args.command == "projects":
        return client.projects()
    if args.command == "catalog":
        return client.catalog()
    if args.command == "graph":
        return client.graph(args.project_id)
    if args.command == "preview-patch":
        return client.preview_patch(args.project_id, _load_object(args.path))
    if args.command == "apply-patch":
        return client.apply_patch(args.project_id, _load_object(args.path))
    if args.command == "validate":
        project = _object(client.graph(args.project_id), label="project")
        document = _object(project.get("document"), label="project document")
        return client.validate(args.project_id, document)
    if args.command == "deploy":
        project = _object(client.graph(args.project_id), label="project")
        document = _object(project.get("document"), label="project document")
        revision = document.get("graphRevision")
        if not isinstance(revision, int):
            raise ValueError("project document has no integer graphRevision")
        return client.deploy(args.project_id, graph_revision=revision, request_id=f"cli:{uuid4().hex}")
    if args.command == "monitors":
        return client.monitors(args.project_id)
    if args.command == "agent-providers":
        return client.agent_providers()
    if args.command == "agent-run":
        return _run_agent(client, args)
    if args.command == "agent-session":
        return client.agent_session(args.session_id)
    if args.command == "agent-cancel":
        return client.cancel_agent_run(args.session_id)
    raise ValueError(f"unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        with StudioApiClient(args.url) as client:
            _print(_dispatch(client, args))
    except (StudioApiError, OSError, ValueError, TimeoutError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
