from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit
from typing import cast

from starlette.datastructures import Headers
from starlette.responses import HTMLResponse, JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

COOKIE_NAME = "f8studio_access"


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def origin_of(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Studio origin must be an HTTP(S) origin without credentials")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return f"{parsed.scheme}://{host}:{port}"


@dataclass(frozen=True)
class StudioAccess:
    token: str
    origins: frozenset[str]

    @classmethod
    def create(cls, data_dir: Path, *, origins: tuple[str, ...]) -> StudioAccess:
        access = cls(secrets.token_urlsafe(32), frozenset(origin_of(origin) for origin in origins))
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "access.json"
        # Replace atomically; readers never observe a partially written credential.
        temporary = data_dir / f".access-{secrets.token_hex(8)}.json"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump({"token": access.token, "origins": sorted(access.origins)}, output)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return access

    def permits_origin(self, origin: str) -> bool:
        try:
            parsed = urlsplit(origin)
            return parsed.path in {"", "/"} and not parsed.query and not parsed.fragment and origin_of(origin) in self.origins
        except ValueError:
            return False


class StudioAccessMiddleware:
    """Authenticate HTTP and WebSocket before routes, including read-only secrets."""

    def __init__(self, app: ASGIApp, *, access: StudioAccess) -> None:
        self.app = app
        self.access = access

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        origin = headers.get("origin")
        if origin is not None and not self.access.permits_origin(origin):
            await self._reject(scope, receive, send, 403, "Origin is not allowed")
            return
        authorization = headers.get("authorization", "")
        token = authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
        if not token:
            from http.cookies import SimpleCookie
            cookies = SimpleCookie()
            cookies.load(headers.get("cookie", ""))
            cookie = cookies.get(COOKIE_NAME)
            token = "" if cookie is None else cookie.value
        authenticated = secrets.compare_digest(token.encode("utf-8"), self.access.token.encode("utf-8"))
        path = scope.get("path", "")
        # Local browser bootstrap only, never a remote or cross-site token endpoint.
        host = urlsplit(f"http://{headers.get('host', '')}").hostname or ""
        peer = scope.get("client")
        local = peer is not None and _loopback(peer[0]) and _loopback(host)
        bootstrap = (scope["type"] == "http" and scope.get("method") == "GET" and path == "/"
                     and local and headers.get("sec-fetch-site", "none") in {"none", "same-origin"})
        if bootstrap:
            async def send_cookie(message: Message) -> None:
                if message["type"] == "http.response.start":
                    cookie = f"{COOKIE_NAME}={self.access.token}; Path=/; HttpOnly; SameSite=Strict"
                    if scope.get("scheme") == "https":
                        cookie += "; Secure"
                    message["headers"] = [*message.get("headers", []), (b"set-cookie", cookie.encode("ascii")),
                                          (b"cache-control", b"no-store")]
                await send(message)
            await self.app(scope, receive, send_cookie)
            return
        if scope["type"] == "http" and path == "/api/auth/session" and scope.get("method") == "POST" and authenticated:
            response = JSONResponse({"authenticated": True})
            response.set_cookie(COOKIE_NAME, self.access.token, httponly=True, samesite="strict",
                                secure=scope.get("scheme") == "https")
            response.headers["Cache-Control"] = "no-store"
            await response(scope, receive, send)
            return
        if not authenticated and scope["type"] == "http" and path == "/" and scope.get("method") == "GET":
            await HTMLResponse(_LOGIN_PAGE, headers={"Cache-Control": "no-store"})(scope, receive, send)
            return
        if not authenticated:
            await self._reject(scope, receive, send, 401, "Studio access token required")
            return
        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send, status: int, message: str) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008, "reason": message})
        else:
            await JSONResponse({"code": "access_denied", "message": message}, status_code=status)(scope, receive, send)


def client_access_token(base_url: str) -> str | None:
    """Do not forward a local credential to an unrelated user-supplied URL."""
    explicit = os.environ.get("F8STUDIO_ACCESS_TOKEN", "").strip()
    server_url = os.environ.get("F8STUDIO_SERVER_URL", "")
    if explicit and server_url and origin_of(base_url) == origin_of(server_url):
        return explicit
    configured = os.environ.get("F8STUDIO_DATA_DIR", "").strip()
    root = Path(configured).expanduser() if configured else Path.home() / ".local/share/f8studio-web"
    path = root / "access.json"
    if not path.exists():
        return None
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Studio access file must contain an object")
    data = cast(dict[str, object], payload)
    token = data.get("token")
    origins = data.get("origins")
    if isinstance(token, str) and isinstance(origins, list) and origin_of(base_url) in cast(list[object], origins):
        return token
    return None

_LOGIN_PAGE = """<!doctype html><meta charset="utf-8"><title>Feel8 Studio sign in</title>
<h1>Feel8 Studio</h1><form><label>Access token <input type="password" autocomplete="off" required></label>
<button>Sign in</button><p role="status"></p></form><script>
document.querySelector('form').onsubmit = async (event) => {
  event.preventDefault();
  const response = await fetch('/api/auth/session', {method: 'POST', headers: {
    Authorization: 'Bearer ' + document.querySelector('input').value}});
  if (response.ok) location.reload();
  else document.querySelector('[role=status]').textContent = 'Sign in failed';
};</script>"""


def client_access_headers(base_url: str) -> dict[str, str]:
    token = client_access_token(base_url)
    return {} if token is None else {"Authorization": f"Bearer {token}"}
