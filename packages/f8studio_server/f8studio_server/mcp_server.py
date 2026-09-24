from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from .api_client import StudioApiClient


def create_mcp_server(
    *,
    studio_url: str,
    host: str,
    port: int,
    path: str,
    client_factory: Callable[[str], StudioApiClient] = StudioApiClient,
) -> Any:
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(
        "f8studio",
        host=host,
        port=port,
        streamable_http_path=path if path.startswith("/") else f"/{path}",
    )

    @server.tool()
    def studio_health() -> object:
        with client_factory(studio_url) as client:
            return client.health()

    @server.tool()
    def project_list() -> object:
        with client_factory(studio_url) as client:
            return client.projects()

    @server.tool()
    def node_catalog() -> object:
        with client_factory(studio_url) as client:
            return client.catalog()

    @server.tool()
    def graph_read(project_id: str) -> object:
        with client_factory(studio_url) as client:
            return client.graph(project_id)

    @server.tool()
    def graph_preview_patch(project_id: str, patch: dict[str, Any]) -> object:
        with client_factory(studio_url) as client:
            return client.preview_patch(project_id, patch)

    @server.tool()
    def graph_apply_patch(project_id: str, patch: dict[str, Any]) -> object:
        with client_factory(studio_url) as client:
            return client.apply_patch(project_id, patch)

    @server.tool()
    def graph_validate(project_id: str, document: dict[str, Any]) -> object:
        with client_factory(studio_url) as client:
            return client.validate(project_id, document)

    @server.tool()
    def project_deploy(project_id: str, graph_revision: int) -> object:
        with client_factory(studio_url) as client:
            return client.deploy(
                project_id,
                graph_revision=graph_revision,
                request_id=f"mcp:{uuid4().hex}",
            )

    @server.tool()
    def runtime_observe(project_id: str | None = None) -> object:
        with client_factory(studio_url) as client:
            return client.monitors(project_id)

    @server.tool()
    def agent_session_create(
        project_id: str,
        title: str = "MCP agent session",
        provider_id: str = "deterministic",
        model_id: str = "graph-builder-v1",
    ) -> object:
        with client_factory(studio_url) as client:
            return client.create_agent_session(
                project_id=project_id,
                title=title,
                provider_id=provider_id,
                model_id=model_id,
            )

    @server.tool()
    def agent_run_start(session_id: str, prompt: str) -> object:
        with client_factory(studio_url) as client:
            return client.start_agent_run(session_id, prompt=prompt)

    @server.tool()
    def agent_session_get(session_id: str) -> object:
        with client_factory(studio_url) as client:
            return client.agent_session(session_id)

    @server.tool()
    def agent_run_cancel(session_id: str) -> object:
        with client_factory(studio_url) as client:
            return client.cancel_agent_run(session_id)

    @server.tool()
    def agent_approval_resolve(
        session_id: str,
        approval_id: str,
        arguments_hash: str,
        approved: bool,
    ) -> object:
        with client_factory(studio_url) as client:
            return client.resolve_agent_approval(
                session_id,
                approval_id,
                approved=approved,
                arguments_hash=arguments_hash,
            )

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Feel8 Web Studio MCP server")
    parser.add_argument("--studio-url", default="http://127.0.0.1:8260")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--path", default="/mcp")
    args = parser.parse_args(argv)
    server = create_mcp_server(
        studio_url=args.studio_url,
        host=args.host,
        port=args.port,
        path=args.path,
    )
    server.run("streamable-http")


if __name__ == "__main__":
    main()
