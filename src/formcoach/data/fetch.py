"""Download, verify and register the public datasets (Checkpoint 1).

``formcoach data fetch --dataset mmfit|recofit|recgym`` calls :func:`fetch_dataset`, which for
every :class:`RemoteFile` in :data:`REGISTRY`:

1. downloads it into ``data/external/<dataset>/`` (resumable: a ``<name>.partial`` file is
   continued with an HTTP ``Range`` request),
2. checks the byte size and the SHA-256 (or MD5 for Zenodo video, which publishes MD5s),
3. extracts zip archives once (marker file ``.<name>.extracted``),
4. upserts a row in ``data/MANIFEST.md``.

Nothing here ever deletes a file (CLAUDE.md rule 7): a mismatching file is reported with the
command to remove it by hand. Checksums below were computed from the files downloaded on
2026-09-24; sizes are exact byte counts.

Sources worth knowing (see docs/04-datasets.md):

* MM-Fit sensor + pose: one 1.74 GB zip on the authors' S3 bucket. Video is separate, per
  workout, on Zenodo record 7607736 (``--with-video --session w00``).
* RecoFit: the two ``.mat`` files are Git LFS objects, so they must come from
  ``media.githubusercontent.com``; the text files come from ``raw.githubusercontent.com``.
* RecGym: the UCI archive (``archive.ics.uci.edu/static/public/1128/...zip``) is served with
  its first 60 MB zeroed (checked 2026-09-24, identical bytes on re-download), so the
  registry uses the authors' Kaggle mirror, which redirects to a signed download without
  credentials. The UCI URL is kept as ``alt_urls`` for the record.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

import requests

log = logging.getLogger(__name__)

CHUNK = 8 * 1024 * 1024
TIMEOUT = 60
ZENODO_RECORD = "7607736"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
MANIFEST_SECTION = "External datasets"


class IntegrityError(RuntimeError):
    """A file on disk does not match the registry (size or checksum)."""


@dataclass(frozen=True)
class RemoteFile:
    """One downloadable file: where it comes from, how big it is, how to verify it."""

    dataset: str
    name: str  # file name under data/external/<dataset>/
    url: str
    size: int | None
    sha256: str | None
    license: str
    extract: bool = False  # zip: extract into the dataset directory
    md5: str | None = None  # Zenodo publishes md5, not sha256
    notes: str = ""
    alt_urls: tuple[str, ...] = field(default_factory=tuple)
    manifest_dataset: str | None = None  # row label when it differs (mmfit-video)


@dataclass
class FetchResult:
    spec: RemoteFile
    path: Path
    status: str  # downloaded | verified | missing
    extracted: bool = False
    sha256: str | None = None


_MMFIT_URL = "https://s3.eu-west-2.amazonaws.com/vradu.uk/mm-fit.zip"
_RECOFIT_MEDIA = (
    "https://media.githubusercontent.com/media/microsoft/"
    "Exercise-Recognition-from-Wearable-Sensors/main/"
)
_RECOFIT_RAW = (
    "https://raw.githubusercontent.com/microsoft/Exercise-Recognition-from-Wearable-Sensors/main/"
)
_RECOFIT_LICENSE = "CDLA-Permissive-2.0 (LICENSE in repo); cite Morris et al. CHI 2014"
_RECGYM_KAGGLE = (
    "https://www.kaggle.com/api/v1/datasets/download/zhaxidelebsz/"
    "10-gym-exercises-with-615-abstracted-features"
)
_RECGYM_UCI = (
    "https://archive.ics.uci.edu/static/public/1128/"
    "recgym:+gym+workouts+recognition+dataset+with+imu+and+capacitive+sensor-7.zip"
)

REGISTRY: dict[str, list[RemoteFile]] = {
    "mmfit": [
        RemoteFile(
            dataset="mmfit",
            name="mm-fit.zip",
            url=_MMFIT_URL,
            size=1_742_309_258,
            sha256="365bf1e546feb9350e69ba026c060fa14c1bccc989d316d062815b4981247b60",
            license="MIT (sensor/pose/code)",
            extract=True,
            notes="21 workouts w00-w20; extracts to mm-fit/; video is separate (Zenodo)",
        )
    ],
    "recofit": [
        RemoteFile(
            dataset="recofit",
            name="exercise_data.50.0000_multionly.mat",
            url=_RECOFIT_MEDIA + "exercise_data.50.0000_multionly.mat",
            size=1_571_881_721,
            sha256="d14a8fa3a6ddb6740ff09f7aa4a3039d3ee5524513a8d4b65ac66e00d14ab509",
            license=_RECOFIT_LICENSE,
            notes="Git LFS object; MATLAB v5 (scipy); 94 subjects, 126 visits with labels",
        ),
        RemoteFile(
            dataset="recofit",
            name="exercise_data.50.0000_singleonly.mat",
            url=_RECOFIT_MEDIA + "exercise_data.50.0000_singleonly.mat",
            size=1_562_300_284,
            sha256="70eea039362555daccb00065e3775ce37fff421f18d533c60933a9d859f33f8a",
            license=_RECOFIT_LICENSE,
            notes="Git LFS object; MATLAB v5 (scipy); 4687 single-exercise recordings",
        ),
        RemoteFile(
            dataset="recofit",
            name="load_exercise_data.m",
            url=_RECOFIT_RAW + "load_exercise_data.m",
            size=6094,
            sha256="6eac24a140518b2ea17983689d71437da731d1fe33907ebb18951f5396517e96",
            license=_RECOFIT_LICENSE,
            notes="format walkthrough",
        ),
        RemoteFile(
            dataset="recofit",
            name="readme.html",
            url=_RECOFIT_RAW + "readme.html",
            size=20420,
            sha256="f347906729d16bad5743501146c1a1fe8d1339f93b059285e375d8e77f37fc71",
            license=_RECOFIT_LICENSE,
        ),
        RemoteFile(
            dataset="recofit",
            name="README.md",
            url=_RECOFIT_RAW + "README.md",
            size=2315,
            sha256="b211c6d5cffaab7a310697b18d8bbbf34324af9777ee4549ae215c1ae624d3e9",
            license=_RECOFIT_LICENSE,
        ),
        RemoteFile(
            dataset="recofit",
            name="LICENSE",
            url=_RECOFIT_RAW + "LICENSE",
            size=2370,
            sha256="9d242f2775d7a1d61249e49ed08bfcca3d8759782ad4bfdaa6dcfff830765054",
            license=_RECOFIT_LICENSE,
        ),
    ],
    "recgym": [
        RemoteFile(
            dataset="recgym",
            name="recgym_kaggle.zip",
            url=_RECGYM_KAGGLE,
            size=116_851_172,
            sha256="0abc140fd2e60ef0edb25a97656eb78a6fbe12c83318fa367646b1b0811e6f92",
            license="CC BY 4.0",
            extract=True,
            alt_urls=(_RECGYM_UCI,),
            notes=(
                "Kaggle mirror (RecGym.csv, 475,013,586 B, 4,703,320 rows, min-max normalised); "
                "the UCI zip is served corrupt (first 60 MB zero) as of 2026-09-24"
            ),
        )
    ],
}


def external_root(repo_root: Path | None = None) -> Path:
    """``data/external`` under the repository root (default: the checkout containing this file)."""
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "data" / "external"


def default_manifest_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "data" / "MANIFEST.md"


def sha256_of(path: Path, algorithm: str = "sha256") -> str:
    """Hex digest of a file, streamed in 8 MB chunks (``algorithm`` may be ``md5``)."""
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_size: int | None, progress: bool) -> None:
    """Stream ``url`` into ``dest``'s ``.partial`` sibling, resuming, then rename to ``dest``."""
    partial = dest.with_name(dest.name + ".partial")
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT, allow_redirects=True) as r:
        if have and r.status_code == 200:  # server ignored Range: start over
            have = 0
        elif have and r.status_code != 206:
            r.raise_for_status()
        r.raise_for_status()
        total = expected_size
        if total is None and r.headers.get("Content-Length"):
            total = have + int(r.headers["Content-Length"])
        bar = None
        if progress:
            try:
                from tqdm import tqdm

                bar = tqdm(
                    total=total, initial=have, unit="B", unit_scale=True, desc=dest.name, leave=False
                )
            except ImportError:  # pragma: no cover
                bar = None
        mode = "ab" if have else "wb"
        with partial.open(mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                if bar:
                    bar.update(len(chunk))
        if bar:
            bar.close()
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise IntegrityError(
            f"{partial}: size {partial.stat().st_size:,} != expected {expected_size:,} after "
            "download; re-run to resume, or delete the .partial file by hand and retry"
        )
    partial.replace(dest)


def _check(path: Path, spec: RemoteFile) -> str:
    """Verify size and checksum of ``path``; return the sha256 hex digest."""
    size = path.stat().st_size
    if spec.size is not None and size != spec.size:
        raise IntegrityError(
            f"{path}: size {size:,} != expected {spec.size:,}. The file is not deleted "
            f"automatically (rule 7); remove it by hand and re-run `formcoach data fetch`."
        )
    digest = sha256_of(path)
    if spec.sha256 and digest != spec.sha256:
        raise IntegrityError(
            f"{path}: sha256 {digest} != expected {spec.sha256}. Remove the file by hand and "
            "re-run `formcoach data fetch`."
        )
    if spec.md5 and (md5 := sha256_of(path, "md5")) != spec.md5:
        raise IntegrityError(f"{path}: md5 {md5} != expected {spec.md5}")
    return digest


def extract_zip(archive: Path, dest_dir: Path) -> bool:
    """Extract ``archive`` into ``dest_dir`` unless the ``.<name>.extracted`` marker exists."""
    marker = dest_dir / f".{archive.name}.extracted"
    if marker.exists():
        return False
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad is not None:
            raise IntegrityError(f"{archive}: corrupt member {bad!r}")
        z.extractall(dest_dir)
    marker.write_text(date.today().isoformat() + "\n", encoding="utf-8")
    return True


def _manifest_row(spec: RemoteFile, digest: str, by: str | None) -> dict[str, str]:
    return {
        "Dataset": spec.manifest_dataset or spec.dataset,
        "File": spec.name,
        "Source URL": spec.url,
        "SHA-256": digest,
        "Size": f"{spec.size:,}" if spec.size else "",
        "License": spec.license,
        "Fetched": date.today().isoformat(),
        "By": by or "",
        "Notes": spec.notes,
    }


def fetch_file(
    spec: RemoteFile,
    root: Path,
    *,
    manifest_path: Path | None = None,
    by: str | None = None,
    verify_only: bool = False,
    progress: bool = False,
) -> FetchResult:
    """Ensure one registry file is present, complete and verified under ``root/<dataset>/``.

    Returns a :class:`FetchResult` whose ``status`` is ``downloaded``, ``verified`` (already
    present) or ``missing`` (``verify_only`` and absent). Raises :class:`IntegrityError` when
    the file on disk does not match; the file is left in place (a fresh download stays as
    ``<name>.partial``).
    """
    dest = root / spec.dataset / spec.name
    status = "verified"
    if not dest.exists():
        if verify_only:
            return FetchResult(spec, dest, "missing")
        partial = dest.with_name(dest.name + ".partial")
        have = partial.stat().st_size if partial.exists() else 0
        log.info("downloading %s (%s)%s", spec.url, spec.name, " resuming" if have else "")
        _download(spec.url, dest, spec.size, progress)
        status = "downloaded"
        try:
            digest = _check(dest, spec)
        except IntegrityError:
            dest.replace(dest.with_name(dest.name + ".partial"))
            raise
    else:
        digest = _check(dest, spec)
    extracted = False
    if spec.extract:
        extracted = extract_zip(dest, dest.parent)
    if manifest_path is not None:
        from formcoach.data import manifest

        manifest.upsert_row(
            manifest_path, MANIFEST_SECTION, ("Dataset", "File"), _manifest_row(spec, digest, by)
        )
    return FetchResult(spec, dest, status, extracted, digest)


def zenodo_video_files(
    sessions: Iterable[str],
    record: dict[str, tuple[str, int]] | None = None,
    modality: str = "rgb",
) -> list[RemoteFile]:
    """Build :class:`RemoteFile` specs for MM-Fit video sessions from the Zenodo record.

    ``record`` maps file name → (checksum string like ``"md5:..."``, size); it is fetched from
    the Zenodo API when omitted. Only ``wXX_<modality>.mp4`` files for the requested sessions
    are returned. RGB files are 0.4–3.1 GB each; ``w00_rgb.mp4`` is 2,169,262,910 bytes.
    """
    if record is None:
        record = fetch_zenodo_record()
    out = []
    for s in sessions:
        name = f"{s}_{modality}.mp4"
        if name not in record:
            raise KeyError(f"{name} not in Zenodo record {ZENODO_RECORD}")
        checksum, size = record[name]
        algo, _, digest = checksum.partition(":")
        out.append(
            RemoteFile(
                dataset="mmfit",
                manifest_dataset="mmfit-video",
                name=name,
                url=f"{ZENODO_API}/files/{name}/content",
                size=size,
                sha256=None,
                md5=digest if algo == "md5" else None,
                license="CC BY 4.0",
                notes="Zenodo 7607736; md5 from the record API",
            )
        )
    return out


def fetch_zenodo_record() -> dict[str, tuple[str, int]]:
    """``{file_name: (checksum, size)}`` for every file in the MM-Fit video record."""
    with urllib.request.urlopen(ZENODO_API, timeout=TIMEOUT) as resp:  # noqa: S310
        data = json.load(resp)
    return {f["key"]: (f["checksum"], int(f["size"])) for f in data["files"]}


def fetch_dataset(
    dataset: str,
    root: Path | None = None,
    *,
    with_video: bool = False,
    sessions: Iterable[str] = ("w00",),
    manifest_path: Path | None = None,
    by: str | None = None,
    verify_only: bool = False,
    progress: bool = True,
) -> list[FetchResult]:
    """Fetch/verify every file of ``dataset`` (and optional MM-Fit video sessions)."""
    if dataset not in REGISTRY:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(REGISTRY)}")
    root = root or external_root()
    specs = list(REGISTRY[dataset])
    if with_video:
        if dataset != "mmfit":
            raise ValueError("--with-video applies to mmfit only")
        specs += [replace(f, dataset="mmfit") for f in zenodo_video_files(sessions)]
    return [
        fetch_file(
            s, root, manifest_path=manifest_path, by=by, verify_only=verify_only, progress=progress
        )
        for s in specs
    ]
