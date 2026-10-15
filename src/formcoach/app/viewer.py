"""Serve the browser BLE live viewer (``web/``) on http://localhost.

Web Bluetooth only runs in a secure context: https (the GitHub Pages copy) or localhost (this
server). Standard library only, so it works from a plain ``make setup`` install.
"""

from __future__ import annotations

import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[3] / "web"
VIEWER_PATH = "/ble-viewer/"


class _Handler(SimpleHTTPRequestHandler):
    # Explicit types: on Windows, mimetypes reads the registry, which can map .js to
    # text/plain, and browsers refuse to run module scripts served that way.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".html": "text/html",
    }

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")  # always serve the files on disk
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass  # one line per request would bury the URL the user needs


def make_server(port: int = 8000, root: Path = WEB_ROOT) -> ThreadingHTTPServer:
    """Serve ``root`` on 127.0.0.1:``port`` (0 picks a free port). Call ``serve_forever()``."""
    handler = functools.partial(_Handler, directory=str(root))
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def viewer_url(server: ThreadingHTTPServer) -> str:
    """The viewer's URL. ``localhost`` (a secure context) is what WSL2 forwards to Windows."""
    return f"http://localhost:{server.server_address[1]}{VIEWER_PATH}"
