"""`formcoach data fetch`: resumable download, hash verification, extraction, manifest rows.

Uses a local HTTP server with Range support so no test touches the network.
"""

from __future__ import annotations

import hashlib
import io
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from formcoach.data import fetch, manifest

PAYLOAD = bytes(range(256)) * 400  # 102,400 bytes


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("inner/data.csv", "a,b\n1,2\n")
    return buf.getvalue()


ZIP = _zip_bytes()
FILES = {"/payload.bin": PAYLOAD, "/archive.zip": ZIP}


class _RangeHandler(BaseHTTPRequestHandler):
    ranges: list[str] = []

    def log_message(self, *_):  # silence
        pass

    def do_HEAD(self):
        self._serve(head=True)

    def do_GET(self):
        self._serve(head=False)

    def _serve(self, head: bool):
        body = FILES.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        rng = self.headers.get("Range")
        start = 0
        if rng:
            _RangeHandler.ranges.append(rng)
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        else:
            self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        if not head:
            self.wfile.write(body[start:])


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _spec(server, name="payload.bin", **kw) -> fetch.RemoteFile:
    body = FILES["/" + name]
    defaults = dict(
        dataset="demo",
        name=name,
        url=f"{server}/{name}",
        size=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
        license="test",
    )
    defaults.update(kw)
    return fetch.RemoteFile(**defaults)


def test_fresh_download_verifies_and_writes_manifest(server, tmp_path):
    spec = _spec(server)
    man = tmp_path / "MANIFEST.md"
    man.write_text(
        "## External datasets\n\n| Dataset | File | Source URL | SHA-256 | Size | License |"
        " Fetched | By | Notes |\n|---|---|---|---|---|---|---|---|---|\n",
        encoding="utf-8",
    )
    result = fetch.fetch_file(spec, tmp_path / "data", manifest_path=man, by="pytest")
    assert result.path.read_bytes() == PAYLOAD
    assert result.status == "downloaded"
    rows = manifest.read_rows(man, "External datasets")
    assert rows[0]["File"] == "payload.bin" and rows[0]["SHA-256"] == spec.sha256
    assert rows[0]["Size"] == f"{len(PAYLOAD):,}"
    # second call: nothing to download, still verified
    again = fetch.fetch_file(spec, tmp_path / "data", manifest_path=man, by="pytest")
    assert again.status == "verified"
    assert len(manifest.read_rows(man, "External datasets")) == 1


def test_resume_from_partial_file_uses_range(server, tmp_path):
    spec = _spec(server)
    dest = tmp_path / "data" / "demo" / "payload.bin"
    dest.parent.mkdir(parents=True)
    partial = dest.with_name(dest.name + ".partial")
    partial.write_bytes(PAYLOAD[:40_000])
    _RangeHandler.ranges.clear()
    result = fetch.fetch_file(spec, tmp_path / "data")
    assert result.status == "downloaded"
    assert result.path.read_bytes() == PAYLOAD
    assert _RangeHandler.ranges == ["bytes=40000-"]
    assert not partial.exists()


def test_bad_hash_raises_and_keeps_partial(server, tmp_path):
    spec = _spec(server, sha256="00" * 32)
    with pytest.raises(fetch.IntegrityError, match="sha256"):
        fetch.fetch_file(spec, tmp_path / "data")
    dest = tmp_path / "data" / "demo" / "payload.bin"
    assert not dest.exists()
    assert dest.with_name(dest.name + ".partial").exists()


def test_existing_file_with_wrong_size_is_reported_not_deleted(server, tmp_path):
    spec = _spec(server)
    dest = tmp_path / "data" / "demo" / "payload.bin"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(PAYLOAD + b"extra")
    with pytest.raises(fetch.IntegrityError, match="size"):
        fetch.fetch_file(spec, tmp_path / "data")
    assert dest.exists()  # rule 7: never delete data


def test_zip_is_extracted_once(server, tmp_path):
    spec = _spec(server, name="archive.zip", extract=True)
    result = fetch.fetch_file(spec, tmp_path / "data")
    assert (tmp_path / "data" / "demo" / "inner" / "data.csv").read_text() == "a,b\n1,2\n"
    assert result.extracted
    marker = tmp_path / "data" / "demo" / ".archive.zip.extracted"
    assert marker.exists()
    result2 = fetch.fetch_file(spec, tmp_path / "data")
    assert not result2.extracted  # skipped


def test_verify_only_reports_missing(server, tmp_path):
    spec = _spec(server)
    result = fetch.fetch_file(spec, tmp_path / "data", verify_only=True)
    assert result.status == "missing"


def test_registry_is_complete():
    for dataset, files in fetch.REGISTRY.items():
        assert files, dataset
        for f in files:
            assert f.url.startswith("https://"), f
            assert f.size and f.sha256 and len(f.sha256) == 64, f
            assert f.license, f
    assert set(fetch.REGISTRY) == {"mmfit", "recofit", "recgym"}
    assert fetch.REGISTRY["mmfit"][0].size == 1_742_309_258
    names = {f.name for f in fetch.REGISTRY["recofit"]}
    assert {"exercise_data.50.0000_multionly.mat", "exercise_data.50.0000_singleonly.mat"} <= names
    # RecoFit .mat files are Git LFS objects: they must come from the media host
    for f in fetch.REGISTRY["recofit"]:
        host = (
            "media.githubusercontent.com"
            if f.name.endswith(".mat")
            else "raw.githubusercontent.com"
        )
        assert host in f.url, f


def test_zenodo_video_spec_builder():
    files = fetch.zenodo_video_files(
        ["w00"], record={"w00_rgb.mp4": ("md5:faa0", 2169262910), "w00_depth.mp4": ("md5:74b3", 1)}
    )
    assert [f.name for f in files] == ["w00_rgb.mp4"]
    assert files[0].md5 == "faa0" and files[0].size == 2169262910
    assert files[0].url.endswith("/files/w00_rgb.mp4/content")
