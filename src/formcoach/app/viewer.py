"""Serve the browser BLE live viewer (``web/``) on http://localhost.

Web Bluetooth only runs in a secure context: https (the GitHub Pages copy) or localhost (this
server). Standard library only, so it works from a plain ``make setup`` install.
"""

from __future__ import annotations

import functools
import os
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


class _ExclusiveServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second socket bind a port another program already holds,
    # so a busy port would not raise. Elsewhere it only allows a quick restart (TIME_WAIT).
    allow_reuse_address = False


def _on_windows() -> bool:
    return os.name == "nt"


def make_server(port: int = 8000, root: Path = WEB_ROOT) -> ThreadingHTTPServer:
    """Serve ``root`` on 127.0.0.1:``port`` (0 picks a free port). Call ``serve_forever()``.

    Raises ``OSError`` if the port is already in use.
    """
    handler = functools.partial(_Handler, directory=str(root))
    server_cls = _ExclusiveServer if _on_windows() else ThreadingHTTPServer
    return server_cls(("127.0.0.1", port), handler)


def viewer_url(server: ThreadingHTTPServer) -> str:
    """The viewer's URL. ``localhost`` (a secure context) is what WSL2 forwards to Windows."""
    return f"http://localhost:{server.server_address[1]}{VIEWER_PATH}"
