"""``Hub``: fan the session's messages out to every connected page (Bench design §5.1, §5.3).

Any thread may publish (the session thread, the link worker, the server's own loop). The hub
never touches a client from another thread: :meth:`Hub.publish` and :meth:`Hub.publish_frame`
hand the message to the asyncio loop with ``loop.call_soon_threadsafe`` and the loop puts it in
each client's queues. Publishing therefore never blocks and a slow page never slows the session.

Each :class:`Client` has an ordered queue (state, events, IMU, link and log messages: never
dropped, never reordered) and a one-item latest-frame slot (an older frame not yet sent is
replaced by the newer one). A client whose ordered queue reaches :data:`MAX_QUEUE` messages has
stopped reading: it is dropped from the hub, its queue is freed and its :meth:`Client.next`
raises :class:`ClientOverflowError`, on which the server closes that socket (the page
reconnects and gets a fresh ``hello``). The other clients are not affected.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque

MAX_QUEUE = 10_000  # ordered messages a client may fall behind before it is disconnected


class ClientOverflowError(Exception):
    """The client fell :data:`MAX_QUEUE` messages behind and was dropped."""


class Client:
    """One connected page's queues. Created by :meth:`Hub.register`; every method except
    :meth:`next` runs on the hub's loop thread."""

    def __init__(self, hub: Hub, max_queue: int):
        self._hub = hub
        self._max = max_queue
        self._ordered: deque[dict] = deque()
        self._frame: bytes | None = None
        self._wake = asyncio.Event()
        self.overflowed = False

    def send(self, msg: dict) -> None:
        """Queue ``msg`` for this client only (loop thread only), e.g. an error about a command
        this page sent."""
        if self.overflowed:
            return
        if len(self._ordered) >= self._max:
            self.overflowed = True
            self._ordered.clear()
            self._frame = None
            self._hub.unregister(self)
        else:
            self._ordered.append(msg)
        self._wake.set()

    def _set_frame(self, data: bytes) -> None:
        if not self.overflowed:
            self._frame = data
            self._wake.set()

    async def next(self) -> dict | bytes:
        """The next message to send: queued ordered messages first, then the latest frame.
        Waits until there is one.

        Raises:
            ClientOverflowError: the client fell too far behind and was dropped.
        """
        while True:
            if self.overflowed:
                raise ClientOverflowError(
                    f"more than {self._max} messages queued; the page stopped reading"
                )
            if self._ordered:
                return self._ordered.popleft()
            if self._frame is not None:
                frame, self._frame = self._frame, None
                return frame
            self._wake.clear()
            await self._wake.wait()


class Hub:
    """The set of connected clients and the thread-safe way to reach them (see the module
    doc). ``max_queue`` is each client's ordered-queue cap."""

    def __init__(self, *, max_queue: int = MAX_QUEUE):
        self.max_queue = max_queue
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[Client] = set()

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Deliver on ``loop`` (the server's event loop). :meth:`register` attaches the running
        loop itself if this was not called."""
        self._loop = loop

    def client_count(self) -> int:
        """How many pages are connected. Any thread."""
        return len(self._clients)

    def publish(self, msg: dict) -> None:
        """Queue ``msg`` for every connected client, in publishing order. Thread-safe, never
        blocks; a no-op with no clients."""
        self._post(self._deliver, msg)

    def publish_frame(self, data: bytes) -> None:
        """Make ``data`` (a binary frame) every client's latest frame, replacing one not yet
        sent. Thread-safe, never blocks; a no-op with no clients."""
        self._post(self._deliver_frame, data)

    def _post(self, fn, arg) -> None:
        loop = self._loop
        if loop is None or not self._clients:
            return
        with contextlib.suppress(RuntimeError):  # the loop is closed: the server has stopped
            loop.call_soon_threadsafe(fn, arg)

    def _deliver(self, msg: dict) -> None:
        for client in tuple(self._clients):
            client.send(msg)

    def _deliver_frame(self, data: bytes) -> None:
        for client in tuple(self._clients):
            client._set_frame(data)

    async def register(self) -> Client:
        """Add a client (call on the server's loop); it receives everything published from now
        on."""
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        client = Client(self, self.max_queue)
        self._clients.add(client)
        return client

    def unregister(self, client: Client) -> None:
        """Remove ``client`` (loop thread); nothing more is queued for it."""
        self._clients.discard(client)
