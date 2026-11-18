"""The Bench pages are served whole (Bench plan Tasks 10, 11): every file a page references and all
module its scripts import, exists under the package's ``static/`` or the vendored uPlot
directory, and the lab console loads its styles and uPlot before its module.

No session, camera or radio: the page routes never touch the hub or the controller beyond the
app's lifespan, so a stub hub stands in.
"""

from __future__ import annotations

import re

import pytest

from formcoach.bench.server import STATIC_DIR, VENDOR_DIR

REF = re.compile(r'(?:src|href)="([^"]+)"')
IMPORT = re.compile(r'(?:from|import)\s*\(?\s*"(\./[^"]+)"')


class _Hub:
    """Just enough of :class:`formcoach.bench.hub.Hub` for the app's lifespan."""

    def attach_loop(self, _loop) -> None:
        pass


def _client():
    pytest.importorskip("starlette")
    pytest.importorskip("httpx2")
    from starlette.testclient import TestClient

    from formcoach.bench.server import create_app

    return TestClient(create_app(_Hub(), None), base_url="http://localhost")


PAGES = {
    "lab": [
        "/vendor/uPlot.min.css",
        "/static/common.css",
        "/static/lab.css",
        "/vendor/uPlot.iife.min.js",
        "/static/lab.js",
    ],
    "coach": ["/static/common.css", "/static/coach.css", "/static/coach.js"],
}


@pytest.mark.parametrize("page", sorted(PAGES))
def test_page_serves_and_references_only_files_that_exist(page):
    with _client() as client:
        r = client.get(f"/{page}")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
        refs = REF.findall(r.text)
        assert refs == PAGES[page]
        for ref in refs:
            root, rel = (
                (STATIC_DIR, ref.removeprefix("/static/"))
                if ref.startswith("/static/")
                else (VENDOR_DIR, ref.removeprefix("/vendor/"))
            )
            assert (root / rel).is_file(), f"{ref} is not a file under {root}"
            if root == STATIC_DIR or VENDOR_DIR.is_dir():
                assert client.get(ref).status_code == 200, ref


def test_every_static_module_imports_only_static_files():
    modules = sorted(STATIC_DIR.glob("*.js"))
    assert any(m.name == "lab.js" for m in modules)
    for module in modules:
        for rel in IMPORT.findall(module.read_text(encoding="utf-8")):
            target = (module.parent / rel).resolve()
            assert target.parent == STATIC_DIR.resolve() and target.is_file(), (
                f"{module.name} imports {rel}, which is not a file in static/"
            )
