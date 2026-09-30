from __future__ import annotations

from f8studio_server.errors import InvalidRequestError

import argparse
import os
from ipaddress import ip_address
from pathlib import Path

import uvicorn

from f8media_protocol.client import RemoteMediaGateway, RemoteMediaGatewayConfig

from .app import DEFAULT_ALLOWED_HOSTS, create_app, default_data_dir
from .access import StudioAccess
from .defaults import DEFAULT_STUDIO_PORT
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
    parser.add_argument("--port", default=DEFAULT_STUDIO_PORT, type=int)
    parser.add_argument("--web-dist", type=Path)
    parser.add_argument(
        "--media-gateway-url",
        help="Loopback URL for the Media Gateway. Defaults to a free port when Studio manages the gateway.",
    )
    parser.add_argument(
        "--external-media-gateway",
        action="store_true",
        help="Connect to an already running gateway at --media-gateway-url instead of starting one.",
    )
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
    args = parser.parse_args()
    if args.external_media_gateway and args.media_gateway_url is None:
        parser.error("--external-media-gateway requires --media-gateway-url")
    return args


def main() -> None:
    args = _parse_args()
    host = str(args.host).strip().lower()
    if host != "localhost":
        try:
            bind_address = ip_address(host)
        except ValueError as exc:
            raise InvalidRequestError("Studio host must be localhost or an explicit IP address") from exc
        if bind_address.is_multicast:
            raise InvalidRequestError("Studio host cannot be a multicast address")
    configured_allowed_hosts = {
        str(allowed_host).strip().lower() for allowed_host in args.allowed_host if str(allowed_host).strip()
    }
    if host in {"0.0.0.0", "::"} and not configured_allowed_hosts:
        raise InvalidRequestError("Wildcard Studio binding requires at least one --allowed-host")
    turn_urls = tuple(str(url).strip() for url in args.turn_url if str(url).strip())
    turn_username = None if args.turn_username is None else str(args.turn_username).strip()
    turn_credential = None if args.turn_credential is None else str(args.turn_credential)
    if (turn_username is None) != (turn_credential is None):
        raise InvalidRequestError("--turn-username and --turn-credential must be provided together")
    if args.force_turn and not turn_urls:
        raise InvalidRequestError("--force-turn requires at least one --turn-url")
    for turn_url in turn_urls:
        if not turn_url.startswith(("turn:", "turns:")):
            raise InvalidRequestError(f"TURN URL must use turn: or turns:: {turn_url}")
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
            allowed_hosts = (set(DEFAULT_ALLOWED_HOSTS) - {"testserver"}) | configured_allowed_hosts
            if host not in {"0.0.0.0", "::"}:
                allowed_hosts.add(host)
            access = StudioAccess.create(default_data_dir(), origins=tuple(
                f"http://{'[' + name + ']' if ':' in name else name}:{args.port}" for name in sorted(allowed_hosts)
            ))
            os.environ["F8STUDIO_ACCESS_TOKEN"] = access.token
            app = create_app(
                web_dist=args.web_dist,
                access=access,
                media_gateway=gateway,
                allowed_hosts=tuple(allowed_hosts),
                rtc_configuration=rtc_configuration,
            )
            local_host = "127.0.0.1" if host == "0.0.0.0" else "::1" if host == "::" else host
            url_host = f"[{local_host}]" if ":" in local_host else local_host
            os.environ["F8STUDIO_SERVER_URL"] = f"http://{url_host}:{args.port}"
            uvicorn.run(app, host=host, port=args.port, log_level="info")
    except StudioServerAlreadyRunningError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
