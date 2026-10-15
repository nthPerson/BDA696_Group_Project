"""`formcoach viewer` serves web/ on localhost with JavaScript MIME types browsers accept."""

import threading
from urllib.request import urlopen

import pytest
from typer.testing import CliRunner

from formcoach.app import viewer
from formcoach.app.viewer import WEB_ROOT, make_server, viewer_url
from formcoach.cli import app


def test_server_serves_the_page_and_modules_as_javascript():
    server = make_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = viewer_url(server)
        assert url.startswith("http://localhost:") and url.endswith("/ble-viewer/")
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urlopen(f"{base}/ble-viewer/") as r:
            assert r.status == 200
            assert "FormCoach" in r.read().decode("utf-8")
        # Windows can map .js to text/plain in the registry; module scripts then refuse to load.
        with urlopen(f"{base}/ble-viewer/protocol.js") as r:
            assert r.headers["Content-Type"].startswith("text/javascript")
            assert r.headers["Cache-Control"] == "no-store"
    finally:
        server.shutdown()
        server.server_close()


def test_web_root_is_the_repo_web_dir():
    assert (WEB_ROOT / "ble-viewer" / "protocol.js").is_file()


def test_viewer_command_is_registered():
    result = CliRunner().invoke(app, ["viewer", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output


def test_a_busy_port_is_an_error_not_a_silent_second_server():
    first = make_server(port=0)
    try:
        with pytest.raises(OSError):
            make_server(port=first.server_address[1]).server_close()
    finally:
        first.server_close()


def test_windows_does_not_set_so_reuseaddr(monkeypatch):
    # On Windows SO_REUSEADDR lets a second process bind a port that is already in use, so a
    # busy port would never raise and requests could go to the other program.
    monkeypatch.setattr(viewer, "_on_windows", lambda: True)
    server = make_server(port=0)
    try:
        assert not server.allow_reuse_address
    finally:
        server.server_close()
