# How to: fetch, verify, convert and profile the public datasets

**Owner:** Ruby Rios Ramirez · **Built by:** Claude Code with Robert on 2026-09-24 · **PR:** #3
**Checkpoint:** 1 (`docs/00-START-HERE.md`) · **State:** works on real data (all three datasets)

## 1. Run it (verified 2026-09-24 on Linux/WSL2)

Fetch (or verify files already on disk) and write the manifest rows. Resumable; safe to rerun.
Windows: replace `make …` with the `uv run formcoach …` line shown under it.

```bash
make data DATASET=recofit          # ≈ 3 GB; mmfit is 1.7 GB, recgym 117 MB
uv run formcoach data fetch --dataset recofit
```
```
verified   /home/…/data/external/recofit/exercise_data.50.0000_multionly.mat
verified   /home/…/data/external/recofit/exercise_data.50.0000_singleonly.mat
verified   /home/…/data/external/recofit/load_exercise_data.m
…
manifest: /home/…/data/MANIFEST.md
```

Convert every raw file into IMUStream Parquet (one file per stream) — about 40 s for all three:

```bash
make convert
uv run formcoach data convert
```
```
mmfit: 42 stream file(s) written
recofit: 126 stream file(s) written
recgym: 146 stream file(s) written
streams: 314 under /home/…/data/processed
```

Write the profile report (tables + two figures) that the Written Report Outline cites:

```bash
make profile
uv run formcoach data profile
```
```
wrote /home/…/reports/data_profile.md
```

## 2. Where things live

| Path | What it is |
|---|---|
| `src/formcoach/data/schema.py` | `IMU_COLUMNS`, `CANONICAL_EXERCISES`, `make_imu_stream(...)`, `validate_imu_stream(df)` — the §5 contract |
| `src/formcoach/data/fetch.py` | `REGISTRY` (URL, size, SHA-256 per file), `fetch_dataset(name, root, ...)`, `fetch_file(spec, root, ...)` |
| `src/formcoach/data/manifest.py` | `read_rows(path, section)`, `upsert_row(path, section, key, row)` for `data/MANIFEST.md` |
| `src/formcoach/data/labels.py` | `canonical(dataset, raw)` → `curl/press/raise/squat/other/idle` or `None` (junk); all 75 RecoFit names |
| `src/formcoach/data/mmfit.py` | `load_stream(root, workout, device)`, `load_sets`, `load_pose3d`, `session_clock`, `WORKOUT_SUBJECT`, `SPLITS`, `describe` |
| `src/formcoach/data/recofit.py` | `load_mat`, `list_visits`, `load_stream(root, subject_index, visit)`, `load_segments`, `describe` |
| `src/formcoach/data/recgym.py` | `load_csv`, `list_sessions`, `load_stream(root, subject, position, session)`, `describe` |
| `src/formcoach/data/convert.py` | `convert_dataset(name, raw_root, out_root, force=False)`, `list_streams(out_root)`, `read_stream(path)` |
| `src/formcoach/data/profile.py` | `write_profile(roots, out_md)` → `reports/data_profile.md`, `reports/figures/data_profile_*.png` |
| `src/formcoach/data/fixtures.py` | `build_all()` — rebuilds `data/fixtures/{mmfit,recofit,recgym}` (`formcoach data make-fixtures`) |
| `data/external/<dataset>/` | raw downloads (gitignored); `data/processed/<dataset>/streams/*.parquet` converted streams (gitignored) |
| `data/MANIFEST.md` | one row per raw file: URL, SHA-256, size, license, date |
| `tests/test_schema.py test_manifest.py test_fetch.py test_loaders.py test_convert_profile.py` | the tests |

## 3. How it works

Every loader turns its raw format into the same DataFrame (`docs/02-system-design.md` §5):
`dataset, subject, session, device, placement, t, ax, ay, az, gx, gy, gz, exercise, rep_id,
set_id` plus `units` and a dataset-specific `label_raw`. `t` is float seconds, strictly
increasing per stream; accel m/s²; gyro rad/s. RecoFit is converted from g and dps; MM-Fit is
already SI; RecGym has **no physical units** (min-max normalised), so its streams carry
`units == "normalized"` and are only used within RecGym (ADR-0014). `exercise` is canonical
(ADR-0015); `set_id` numbers labelled sets so `eval repcount` can look up rep counts
(`mmfit.load_sets`, `recofit.load_segments`). `data convert` writes the streams once; nothing
downstream reads `.mat`/`.npy`/CSV again.

## 4. Tests

`uv run pytest tests/test_schema.py tests/test_manifest.py tests/test_fetch.py tests/test_loaders.py tests/test_convert_profile.py`
(51 tests, < 10 s, no network: `test_fetch.py` runs a local HTTP server with Range support).
`test_loaders.py` pins units, label boundaries, missing modalities and the schema on the
fixtures. If you change a loader's output columns, update `schema.py` (add, never rename),
`tests/test_schema.py`, and rerun `make convert` with `--force`. If you change `fixtures.py`,
run `uv run formcoach data make-fixtures` and commit the regenerated files (< 400 KB each).

## 5. Could not verify / open questions

- RecoFit sensor placement is inferred from `masterToken == 'rightArm'` and the CHI paper
  (forearm); the readme does not say. Placement is `forearm_r` (ADR-0016).
- MM-Fit smartwatch axis orientation vs. our wearable is unknown until we record with both.
- The UCI RecGym archive is corrupt on the server (ADR-0013); if UCI fixes it, add its
  SHA-256 to `REGISTRY` and drop the Kaggle note.
- `--with-video` (Zenodo MD5 verification) is implemented and unit-tested against a fake
  record but was **not** run against Zenodo (2.17 GB for `w00_rgb.mp4`).

## 6. Next steps for Ruby

1. Read `reports/data_profile.md`; sanity-check the class-minute table against the papers.
2. Checkpoint 2 consumes `data/processed/*/streams/`; if you add a dataset, follow
   `recgym.py` (smallest loader) and add it to `fetch.REGISTRY`, `labels.py`, `convert.py`.
3. When Robert's team recordings exist, `data/team.py` should emit the same schema from
   `imu.parquet` (Checkpoint 4 writes it).
