"""FormCoach Bench: the local web server behind the lab console and the coach page (Bench design
§5.3).

One process runs a :class:`formcoach.app.session.LiveSession` on a worker thread and serves
browser pages and one WebSocket on ``127.0.0.1``. :mod:`.protocol` is the message schema (pure,
importable without the ``bench`` extra), :mod:`.hub` fans messages out to the connected pages,
:mod:`.controller` turns page commands into session calls and session output into messages,
:mod:`.server` is the Starlette app and :mod:`.runner` assembles everything for
``formcoach bench``. The core (``formcoach.app``, ``io``, ``pose``, ``signal``) never imports
this package.

The constants below are cheap to import (the CLI reads them for its defaults).
"""

VIDEO_WIDTH = 640  # px: frames are resized to this width before JPEG encoding (measured)
JPEG_QUALITY = 75  # 0-100 (measured)
VIDEO_MAX_FPS = 30.0  # images per second at most, on the session clock (measured)
HOST = "127.0.0.1"  # the server binds here only
DEFAULT_HTTP_PORT = 8010
