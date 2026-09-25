import asyncio
import os
import socket
import sys

import httpx
import pytest

from f8media_gateway.app import create_app
from f8media_protocol.client import (
    MediaGatewayRequestError,
    MediaGatewayUnavailable,
    RemoteMediaGateway,
    RemoteMediaGatewayConfig,
)
from f8media_protocol.models import MediaSessionOffer


def unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_remote_client_round_trips_gateway_contract() -> None:
    async def scenario() -> None:
        transport = httpx.ASGITransport(app=create_app())
        client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
        gateway = RemoteMediaGateway(
            RemoteMediaGatewayConfig(base_url="http://testserver", manage_process=False),
            client=client,
        )
        try:
            await gateway.start()
            health = await gateway.health()
            assert health.service == "f8media-gateway"
            assert await gateway.close_video_session("missing") is False
            with pytest.raises(MediaGatewayRequestError) as rejected:
                await gateway.create_video_session(
                    MediaSessionOffer(source="synthetic://bars", quality="invalid", sdp="offer")
                )
            assert rejected.value.status_code == 422
            assert rejected.value.detail == "media quality must be thumbnail or main"
        finally:
            await gateway.close()
            await client.aclose()

    asyncio.run(scenario())


def test_remote_client_rejects_incompatible_protocol() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "service": "f8media-gateway",
                "version": "0.0.1",
                "protocolVersion": "f8media-api/0",
                "gatewayEpoch": "old",
                "processId": 1,
            },
        )

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://testserver")
        gateway = RemoteMediaGateway(
            RemoteMediaGatewayConfig(base_url="http://testserver", manage_process=False),
            client=client,
        )
        try:
            with pytest.raises(MediaGatewayUnavailable, match="protocol mismatch"):
                await gateway.start()
        finally:
            await gateway.close()
            await client.aclose()

    asyncio.run(scenario())


def test_managed_gateway_rejects_health_from_another_process() -> None:
    async def scenario() -> None:
        port = unused_loopback_port()
        occupying_gateway = RemoteMediaGateway(
            RemoteMediaGatewayConfig(
                base_url=f"http://127.0.0.1:{port}",
                manage_process=True,
            )
        )
        conflicting_gateway = RemoteMediaGateway(
            RemoteMediaGatewayConfig(
                base_url=f"http://127.0.0.1:{port}",
                manage_process=True,
            )
        )
        try:
            await occupying_gateway.start()
            with pytest.raises(
                MediaGatewayUnavailable,
                match="process mismatch|exited during startup",
            ):
                await conflicting_gateway.start()
        finally:
            await conflicting_gateway.close()
            await occupying_gateway.close()

    asyncio.run(scenario())


def test_managed_gateway_runs_in_a_separate_process_and_stops() -> None:
    async def scenario() -> None:
        port = unused_loopback_port()
        gateway = RemoteMediaGateway(
            RemoteMediaGatewayConfig(
                base_url=f"http://127.0.0.1:{port}",
                manage_process=True,
                startup_timeout_s=10.0,
            )
        )
        process_id = 0
        try:
            await gateway.start()
            health = await gateway.health()
            process_id = health.process_id
            assert process_id != os.getpid()
            process = gateway._process
            assert process is not None
            assert process.pid == process_id
            assert process.returncode is None
        finally:
            await gateway.close()
        assert process_id > 0
        assert process is not None
        assert process.returncode is not None

    asyncio.run(scenario())


def test_gateway_exits_when_parent_watch_pipe_closes() -> None:
    async def scenario() -> None:
        port = unused_loopback_port()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "f8media_gateway",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--exit-on-stdin-close",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            deadline = asyncio.get_running_loop().time() + 10.0
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
                while True:
                    try:
                        response = await client.get("/api/health")
                    except httpx.ConnectError:
                        if asyncio.get_running_loop().time() >= deadline:
                            raise TimeoutError("gateway did not become ready") from None
                        await asyncio.sleep(0.05)
                        continue
                    assert response.status_code == 200
                    break
            if process.stdin is None:
                raise AssertionError("gateway parent-watch stdin pipe is unavailable")
            process.stdin.close()
            await process.stdin.wait_closed()
            await asyncio.wait_for(process.wait(), timeout=5.0)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    asyncio.run(scenario())
