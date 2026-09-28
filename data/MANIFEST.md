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
| mmfit | mm-fit.zip | https://s3.eu-west-2.amazonaws.com/vradu.uk/mm-fit.zip | 365bf1e546feb9350e69ba026c060fa14c1bccc989d316d062815b4981247b60 | 1,742,309,258 | MIT (sensor/pose/code) | 2026-09-24 | Claude Code (Robert) | 21 workouts w00-w20; extracts to mm-fit/; video is separate (Zenodo) |
| mmfit-video | _pending_ | https://doi.org/10.5281/zenodo.7607736 | | | CC BY 4.0 | | | `--with-video --session w00 w01 w09`; 40.8 GB total, never all of it |
| recofit | exercise_data.50.0000_multionly.mat | https://media.githubusercontent.com/media/microsoft/Exercise-Recognition-from-Wearable-Sensors/main/exercise_data.50.0000_multionly.mat | d14a8fa3a6ddb6740ff09f7aa4a3039d3ee5524513a8d4b65ac66e00d14ab509 | 1,571,881,721 | CDLA-Permissive-2.0 (LICENSE in repo); cite Morris et al. CHI 2014 | 2026-09-24 | Claude Code (Robert) | Git LFS object; MATLAB v5 (scipy); 94 subjects, 126 visits with labels |
| recgym | recgym_kaggle.zip | https://www.kaggle.com/api/v1/datasets/download/zhaxidelebsz/10-gym-exercises-with-615-abstracted-features | 0abc140fd2e60ef0edb25a97656eb78a6fbe12c83318fa367646b1b0811e6f92 | 116,851,172 | CC BY 4.0 | 2026-09-24 | Claude Code (Robert) | Kaggle mirror (RecGym.csv, 475,013,586 B, 4,703,320 rows, min-max normalised); the UCI zip is served corrupt (first 60 MB zero) as of 2026-09-24 |
| recofit | exercise_data.50.0000_singleonly.mat | https://media.githubusercontent.com/media/microsoft/Exercise-Recognition-from-Wearable-Sensors/main/exercise_data.50.0000_singleonly.mat | 70eea039362555daccb00065e3775ce37fff421f18d533c60933a9d859f33f8a | 1,562,300,284 | CDLA-Permissive-2.0 (LICENSE in repo); cite Morris et al. CHI 2014 | 2026-09-24 | Claude Code (Robert) | Git LFS object; MATLAB v5 (scipy); 4687 single-exercise recordings |
| recofit | load_exercise_data.m | https://raw.githubusercontent.com/microsoft/Exercise-Recognition-from-Wearable-Sensors/main/load_exercise_data.m | 6eac24a140518b2ea17983689d71437da731d1fe33907ebb18951f5396517e96 | 6,094 | CDLA-Permissive-2.0 (LICENSE in repo); cite Morris et al. CHI 2014 | 2026-09-24 | Claude Code (Robert) | format walkthrough |
| recofit | readme.html | https://raw.githubusercontent.com/microsoft/Exercise-Recognition-from-Wearable-Sensors/main/readme.html | f347906729d16bad5743501146c1a1fe8d1339f93b059285e375d8e77f37fc71 | 20,420 | CDLA-Permissive-2.0 (LICENSE in repo); cite Morris et al. CHI 2014 | 2026-09-24 | Claude Code (Robert) |  |
| recofit | README.md | https://raw.githubusercontent.com/microsoft/Exercise-Recognition-from-Wearable-Sensors/main/README.md | b211c6d5cffaab7a310697b18d8bbbf34324af9777ee4549ae215c1ae624d3e9 | 2,315 | CDLA-Permissive-2.0 (LICENSE in repo); cite Morris et al. CHI 2014 | 2026-09-24 | Claude Code (Robert) |  |
| recofit | LICENSE | https://raw.githubusercontent.com/microsoft/Exercise-Recognition-from-Wearable-Sensors/main/LICENSE | 9d242f2775d7a1d61249e49ed08bfcca3d8759782ad4bfdaa6dcfff830765054 | 2,370 | CDLA-Permissive-2.0 (LICENSE in repo); cite Morris et al. CHI 2014 | 2026-09-24 | Claude Code (Robert) |  |

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
