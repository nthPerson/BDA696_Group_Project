"""The Bench's Starlette app: the pages, their static files and the one WebSocket (Bench design
§5.3). Needs the ``bench`` extra (``uv sync --extra bench``).

Routes: ``/`` redirects to ``/lab``; ``/lab``, ``/coach`` and ``/review`` are the pages
(``static/<name>.html``); ``/static/*`` serves this package's ``static/`` directory; ``/vendor/*``
serves the repository's ``web/ble-viewer/vendor/`` (uPlot), and answers 404 when that directory
is absent (an installed wheel); ``/ws`` is the WebSocket (:mod:`formcoach.bench.protocol`).
Files are served with ``Cache-Control: no-store`` (always the files on disk) and explicit MIME
types for JavaScript and CSS: on Windows the registry can map ``.js`` to ``text/plain`` and
browsers refuse module scripts served that way (as in :mod:`formcoach.app.viewer`).
"""

from __future__ import annotations

import asyncio
import contextlib
import mimetypes
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from formcoach.bench import protocol
from formcoach.bench.controller import Controller
from formcoach.bench.hub import ClientOverflowError, Hub

STATIC_DIR = Path(__file__).resolve().parent / "static"
VENDOR_DIR = Path(__file__).resolve().parents[3] / "web" / "ble-viewer" / "vendor"
PAGES = ("lab", "coach", "review")
NO_STORE = {"Cache-Control": "no-store"}
MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}
OVERFLOW_CLOSE_CODE = 1013  # "try again later": the page stopped reading; it reconnects


def _serve_file(root: Path, rel: str) -> Response:
    """``root/rel`` with ``no-store`` and an explicit MIME type, or 404 when it does not exist
    or lies outside ``root``."""
    try:
        base = root.resolve(strict=True)
        path = (base / rel).resolve(strict=True)
    except (OSError, RuntimeError):
        return PlainTextResponse("not found", status_code=404, headers=NO_STORE)
    if base not in path.parents or not path.is_file():
        return PlainTextResponse("not found", status_code=404, headers=NO_STORE)
    suffix = path.suffix.lower()
    media = (
        MIME_TYPES.get(suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    )
    return FileResponse(path, media_type=media, headers=NO_STORE)


def create_app(
    hub: Hub,
    controller: Controller,
    *,
    static_dir: Path = STATIC_DIR,
    vendor_dir: Path = VENDOR_DIR,
) -> Starlette:
    """The Bench app (routes in the module doc). ``static_dir``/``vendor_dir`` replace the
    package's ``static/`` and the repository's uPlot directory (tests)."""

    async def root(_request: Request) -> Response:
        return RedirectResponse("/lab", headers=NO_STORE)

    def page(name: str):
        async def endpoint(_request: Request) -> Response:
            return _serve_file(static_dir, f"{name}.html")

        return endpoint

    async def static(request: Request) -> Response:
        return _serve_file(static_dir, request.path_params["path"])

    async def vendor(request: Request) -> Response:
        return _serve_file(vendor_dir, request.path_params["path"])

    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        client = await hub.register()
        try:
            await websocket.send_text(protocol.dumps(protocol.hello(controller.state())))
            reader = asyncio.ensure_future(_read_commands(websocket, client, controller))
            writer = asyncio.ensure_future(_write_messages(websocket, client))
            done, pending = await asyncio.wait({reader, writer}, return_when="FIRST_COMPLETED")
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for task in done:
                with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                    task.result()
        except WebSocketDisconnect:
            pass
        finally:
            hub.unregister(client)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette):
        hub.attach_loop(asyncio.get_running_loop())
        yield

    routes = [
        Route("/", root),
        *(Route(f"/{name}", page(name)) for name in PAGES),
        Route("/static/{path:path}", static),
        Route("/vendor/{path:path}", vendor),
        WebSocketRoute("/ws", ws_endpoint),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


async def _read_commands(websocket: WebSocket, client, controller: Controller) -> None:
    """Parse each text frame as a command and hand it to the controller; a malformed one gets
    an ``error`` sent to this page only. Returns when the page disconnects."""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        text = message.get("text")
        if text is None:
            client.send(protocol.error("commands are JSON text, not binary"))
            continue
        try:
            cmd = protocol.parse_command(text)
        except protocol.CommandError as exc:
            client.send(protocol.error(str(exc), exc.cmd))
            continue
        controller.handle(cmd)


async def _write_messages(websocket: WebSocket, client) -> None:
    """Send the client's queued messages until the socket closes; a client that fell too far
    behind is closed with code :data:`OVERFLOW_CLOSE_CODE`."""
    while True:
        try:
            item = await client.next()
        except ClientOverflowError as exc:
            with contextlib.suppress(Exception):
                await websocket.close(code=OVERFLOW_CLOSE_CODE, reason=str(exc)[:120])
            return
        if isinstance(item, bytes):
            await websocket.send_bytes(item)
        else:
            await websocket.send_text(protocol.dumps(item))
