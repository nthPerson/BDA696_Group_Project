# Data manifest

Every file under `data/external/` (raw downloads) and `data/team/` (team recordings) is listed
here with its source, checksum, size, license and fetch date. Raw data is **never committed**;
this manifest and the small fixtures under `data/fixtures/` are the only data files in git.
`formcoach data fetch` appends rows automatically (Checkpoint 1); until then add rows by hand.

Verify a file: `sha256sum <file>` (Linux), `shasum -a 256 <file>` (macOS),
`Get-FileHash <file> -Algorithm SHA256` (PowerShell).

## External datasets

| Dataset | File | Source URL | SHA-256 | Size | License | Fetched | By | Notes |
|---|---|---|---|---|---|---|---|---|
| mmfit | _pending_ | https://mmfit.github.io/ (sensor + pose zip) | | | MIT (code/sensor/pose) | | | video is separate (Zenodo, CC BY 4.0); fetch selected sessions only |
| mmfit-video | _pending_ | https://doi.org/10.5281/zenodo.7607736 | | | CC BY 4.0 | | | `--with-video --session w00 w01 w09`; 40.8 GB total, never all of it |
| recofit | _pending_ | https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors | | ~2.5 GB | see repo LICENSE; cite Morris et al. CHI 2014 | | | two .mat files + readme.html + load_exercise_data.m |
| recgym | _pending_ | https://archive.ics.uci.edu/dataset/1128 | | 103 MB zip / 453 MB csv | CC BY 4.0 | | | 20 Hz; drop `C_1` |

## Fixtures (committed, small)

| File | Origin | Purpose | Size |
|---|---|---|---|
| _pending_ | | 30-second replay fixture for the integration test and `make demo` | < 2 MB |

## Team recordings (`data/team/<S#>/<session_id>/`, gitignored)

Subjects are coded `S1`–`S5`; only `imu.parquet`, `pose.parquet`, `events.parquet` and
`meta.json` are ever shared. `video.mp4` stays on the recording laptop. Protocol:
`docs/04-datasets.md` §6.

| Subject | Session | Exercise | Device | Recorded | Shared files | Notes |
|---|---|---|---|---|---|---|

## Citations

- Strömbäck, Huang & Radu. *MM-Fit: Multimodal Deep Learning for Automatic Exercise Logging across Sensing Devices.* IMWUT 4(4), 2020. https://doi.org/10.1145/3432701
- Morris, Saponas, Guillory & Kelner. *RecoFit: Using a Wearable Sensor to Find, Recognize, and Count Repetitive Exercises.* CHI 2014. https://doi.org/10.1145/2556288.2557116
- RecGym, UCI Machine Learning Repository, dataset 1128 (CC BY 4.0).
