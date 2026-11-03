"""``landmarker.fetch_model``: a Python (``requests``) download, no real network call here."""

from __future__ import annotations

import pytest

from formcoach.pose import landmarker


class _FakeResponse:
    def __init__(self, chunks, *, ok: bool = True):
        self._chunks = chunks
        self._ok = ok

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("bad status")

    def iter_content(self, chunk_size=None):
        yield from self._chunks


def test_fetch_model_downloads_once(tmp_path):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return _FakeResponse([b"abc", b"def"])

    out = landmarker.fetch_model("lite", dest_dir=tmp_path, get=fake_get)

    assert out == tmp_path / "pose_landmarker_lite.task"
    assert out.read_bytes() == b"abcdef"
    assert not out.with_suffix(out.suffix + ".partial").exists()
    assert len(calls) == 1
    assert calls[0][0] == landmarker.MODEL_URLS["lite"]
    assert calls[0][1].get("stream") is True

    out2 = landmarker.fetch_model("lite", dest_dir=tmp_path, get=fake_get)
    assert out2 == out
    assert len(calls) == 1  # file already existed; get() not called again


def test_fetch_model_rejects_unknown_variant(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        landmarker.fetch_model("xl", dest_dir=tmp_path, get=lambda *a, **k: None)
    assert list(tmp_path.iterdir()) == []


def test_fetch_model_failure_leaves_no_partial_file(tmp_path):
    def failing_get(url, **kwargs):
        def gen():
            yield b"some-bytes"
            raise ConnectionError("boom")

        return _FakeResponse(gen())

    with pytest.raises(ConnectionError):
        landmarker.fetch_model("lite", dest_dir=tmp_path, get=failing_get)

    assert not (tmp_path / "pose_landmarker_lite.task").exists()
    assert not (tmp_path / "pose_landmarker_lite.task.partial").exists()


def test_fetch_model_failure_on_bad_status_leaves_no_partial_file(tmp_path):
    def fake_get(url, **kwargs):
        return _FakeResponse([], ok=False)

    with pytest.raises(RuntimeError, match="bad status"):
        landmarker.fetch_model("full", dest_dir=tmp_path, get=fake_get)

    assert list(tmp_path.iterdir()) == []


def test_fetch_model_defaults_to_models_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(landmarker, "MODELS_DIR", tmp_path)

    def fake_get(url, **kwargs):
        return _FakeResponse([b"xyz"])

    out = landmarker.fetch_model("lite", get=fake_get)
    assert out == tmp_path / "pose_landmarker_lite.task"
    assert out.exists()
