"""Studio server entrypoint; publisher tooling can import contracts without loading the UI."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI
    from f8media_protocol.contracts import MediaGateway
    from .access import StudioAccess
    from .application import StudioApplication
    from .models import BrowserRtcConfiguration
    from .runtime import RuntimeConfig, RuntimeGateway


def create_app(
    *,
    web_dist: Path | None = None,
    data_dir: Path | None = None,
    runtime: RuntimeGateway | None = None,
    runtime_config: RuntimeConfig | None = None,
    service_roots: tuple[Path, ...] | None = None,
    application: StudioApplication | None = None,
    media_gateway: MediaGateway | None = None,
    allowed_hosts: tuple[str, ...] | None = None,
    access: StudioAccess | None = None,
    rtc_configuration: BrowserRtcConfiguration | None = None,
) -> FastAPI:
    from .app import create_app as build_app
    return build_app(web_dist=web_dist, data_dir=data_dir, runtime=runtime, runtime_config=runtime_config,
                     service_roots=service_roots, application=application, media_gateway=media_gateway,
                     allowed_hosts=allowed_hosts, access=access, rtc_configuration=rtc_configuration)


__all__ = ['create_app']
