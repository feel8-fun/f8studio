from __future__ import annotations

import argparse
from ipaddress import ip_address
from pathlib import Path

import uvicorn

from f8media_protocol.client import RemoteMediaGateway, RemoteMediaGatewayConfig

from .app import DEFAULT_ALLOWED_HOSTS, create_app
from .models import BrowserIceServer, BrowserRtcConfiguration
from .server_instance import StudioServerAlreadyRunningError, single_server_instance


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Feel8 Web Studio server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="Trusted HTTP Host/Origin hostname. Repeat for multiple names when binding a wildcard address.",
    )
    parser.add_argument("--port", default=8210, type=int)
    parser.add_argument("--web-dist", type=Path)
    parser.add_argument("--media-gateway-url", default="http://127.0.0.1:8211")
    parser.add_argument("--external-media-gateway", action="store_true")
    parser.add_argument(
        "--turn-url",
        action="append",
        default=[],
        help="Browser TURN URL. Repeat to provide fallback transports.",
    )
    parser.add_argument("--turn-username")
    parser.add_argument("--turn-credential")
    parser.add_argument(
        "--force-turn",
        action="store_true",
        help="Require browser media to use TURN relay candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    host = str(args.host).strip().lower()
    if host != "localhost":
        try:
            bind_address = ip_address(host)
        except ValueError as exc:
            raise ValueError("Studio host must be localhost or an explicit IP address") from exc
        if bind_address.is_multicast:
            raise ValueError("Studio host cannot be a multicast address")
    configured_allowed_hosts = {
        str(allowed_host).strip().lower() for allowed_host in args.allowed_host if str(allowed_host).strip()
    }
    if host in {"0.0.0.0", "::"} and not configured_allowed_hosts:
        raise ValueError("Wildcard Studio binding requires at least one --allowed-host")
    turn_urls = tuple(str(url).strip() for url in args.turn_url if str(url).strip())
    turn_username = None if args.turn_username is None else str(args.turn_username).strip()
    turn_credential = None if args.turn_credential is None else str(args.turn_credential)
    if (turn_username is None) != (turn_credential is None):
        raise ValueError("--turn-username and --turn-credential must be provided together")
    if args.force_turn and not turn_urls:
        raise ValueError("--force-turn requires at least one --turn-url")
    for turn_url in turn_urls:
        if not turn_url.startswith(("turn:", "turns:")):
            raise ValueError(f"TURN URL must use turn: or turns:: {turn_url}")
    ice_server: BrowserIceServer | None = None
    if turn_urls:
        ice_server = BrowserIceServer(
            urls=turn_urls,
            username=turn_username,
            credential=turn_credential,
        )
    rtc_configuration = BrowserRtcConfiguration(
        ice_servers=() if ice_server is None else (ice_server,),
        ice_transport_policy="relay" if args.force_turn else "all",
    )
    try:
        with single_server_instance():
            gateway = RemoteMediaGateway(
                RemoteMediaGatewayConfig(
                    base_url=args.media_gateway_url,
                    manage_process=not args.external_media_gateway,
                )
            )
            allowed_hosts = {*DEFAULT_ALLOWED_HOSTS, *configured_allowed_hosts}
            if host not in {"0.0.0.0", "::"}:
                allowed_hosts.add(host)
            app = create_app(
                web_dist=args.web_dist,
                media_gateway=gateway,
                allowed_hosts=tuple(allowed_hosts),
                rtc_configuration=rtc_configuration,
            )
            uvicorn.run(app, host=host, port=args.port, log_level="info")
    except StudioServerAlreadyRunningError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
