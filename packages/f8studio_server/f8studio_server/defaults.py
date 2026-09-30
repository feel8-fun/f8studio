from __future__ import annotations

# Single source for the Studio server's default listen address. The server,
# the CLI, the MCP bridge and the HTTP client must agree on it.
DEFAULT_STUDIO_PORT = 8210
DEFAULT_STUDIO_URL = f"http://127.0.0.1:{DEFAULT_STUDIO_PORT}"
