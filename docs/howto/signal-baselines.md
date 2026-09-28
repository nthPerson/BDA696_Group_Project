# How to: build windows/features, run LOSO baselines and the rep-count baseline

**Owner:** Christian Byars · **Built by:** Claude Code with Robert on 2026-09-24 · **PR:** #4
**Checkpoint:** 2 (`docs/00-START-HERE.md`) · **State:** works on real data (all reports under `reports/` were produced this way)

## 1. Run it (verified 2026-09-24 on Linux/WSL2, 16 cores)

Windows + 63 features for every converted stream (needs `make convert` first; ~35 s with 14 workers):

```bash
make features                       # = uv run formcoach features build --jobs 4
uv run formcoach features build --jobs 14
```
```
314 window file(s) written
```

Leave-one-subject-out baselines (each writes `reports/loso_<model>_<task>_<dataset>.md` + a
confusion-matrix PNG). MM-Fit has 10 subjects (fast); RecoFit has 94 (RF: ~15–60 min):

```bash
uv run formcoach eval loso --model rf --dataset mmfit --task active
```
```
  fold P0
  …
  fold P9
wrote /home/…/reports/loso_rf_active_mmfit.md
```

Cross-dataset (train on all of RecoFit, test per MM-Fit subject):

```bash
uv run formcoach eval loso --model rf --dataset recofit --task active --test-dataset mmfit
```

Rep-count baseline on the 616 MM-Fit sets × 2 watches (5 s):

```bash
uv run formcoach eval repcount
```
```
exercise  n_sets      mae      bias  exact_pct  within1_pct
    curl     118 0.694915 -0.389831  50.000000    89.830508
   press     120 3.658333  3.491667  17.500000    35.000000
    ...
     all    1232 2.389610  1.715909  40.503247    66.639610
wrote /home/…/reports/baseline_repcount.md
```

## 2. Where things live

| Path | What it is |
|---|---|
| `src/formcoach/signal/resample.py` | `resample_uniform(t, x, fs)`, `resample_stream(df, fs)` (labels by nearest sample) |
| `src/formcoach/signal/filters.py`, `gravity.py` | `lowpass/highpass/bandpass(x, fs, ...)` zero-phase; `split_gravity(acc, fs)`, `magnitude(acc)` |
| `src/formcoach/signal/windows.py` | `make_windows(df, fs, win_s, stride_s, drop_junk)` → Window rows; `windows_to_array(w)` → `(n, 100, 6)` |
| `src/formcoach/signal/features.py` | `window_features(x, fs)` → 63 `feat_*`; `featurize_windows(w, fs)`; `FEATURE_NAMES` |
| `src/formcoach/signal/build.py` | `build_features(processed_root, dataset, ..., jobs)`, `load_windows(root, dataset, min_purity)` |
| `src/formcoach/signal/reps.py` | `count_reps(acc, t, fs, prominence_g, min_distance_s, mode)` → `list[Rep]` |
| `src/formcoach/eval/loso.py`, `loso_cli.py` | `loso_splits`, `group_kfold_splits`, `run_loso(X, y, groups, make_model, folds)`, `write_loso_report`; the CLI glue |
| `src/formcoach/models/energy.py`, `rf.py` | `EnergyGate` (threshold on var\|a\|), `make_rf(seed)` (balanced RF, 200 trees) |
| `src/formcoach/eval/repcount.py` | `count_sets`, `summarize`, `evaluate(...)` → `reports/baseline_repcount.md` (+ `.csv` per set) |
| `data/processed/<dataset>/windows/*.parquet` | one file per stream (gitignored); `reports/loso_*.md`, `reports/transfer_*.md`, `reports/baseline_repcount.md` committed |
| `tests/test_signal.py test_features_build.py test_loso.py test_reps.py` | 30 tests, synthetic signals + fixtures |

## 3. How it works

Streams are resampled to 50 Hz (linear interpolation, duplicate timestamps averaged), cut into
2 s windows every 1 s, labelled by the majority canonical label with a `label_purity` column;
windows overlapping RecoFit junk labels are dropped and `load_windows` keeps purity ≥ 0.8.
`active = label != idle`. Features are 9 statistics × 7 channels (ax..gz, |a|): mean, std,
min, max, energy, dominant frequency (0.3–8 Hz), spectral entropy, autocorrelation peak lag and
height (0.3–3 s). LOSO trains a fresh model per held-out subject and pools predictions for the
confusion matrix; the headline is the pooled unseen-subject macro-F1. RecGym windows are in
normalised units and can only be evaluated within RecGym (`check_units` refuses mixes). The
rep counter band-passes 0.3–3 Hz, picks the axis with most variance and runs
`scipy.signal.find_peaks` (prominence 0.15 g, spacing 0.8 s — docs/02 §4.4 defaults).

## 4. Tests

`uv run pytest tests/test_signal.py tests/test_features_build.py tests/test_loso.py tests/test_reps.py`.
They pin: sine recovery through resampling, 20 Hz upsampling, filter attenuation, gravity
split, window count/labels/purity/junk drop, feature names and dominant frequency, LOSO fold
disjointness, the energy gate and RF on separable synthetic windows, report sections, rep
counts on a clean sine and on noise. Changing `FEATURE_NAMES` or the window layout requires
`make features` with `--force` and regenerating the reports.

## 5. Could not verify / open questions

- The peak counter over-counts slow exercises (press, squat: bias ≈ +3.5 reps per set) with
  the design defaults; `--min-distance-s` and `--prominence-g` are exposed for calibration.
- RecoFit's per-fold RF numbers vary a lot (exercise task min macro-F1 0.27) because several
  subjects have only one or two of the four exercises; the pooled number is the one to cite.
- `eval loso --model cnn` lands in PR 6 (needs `uv sync --extra train`).

## 6. Next steps for Christian

1. Read `reports/loso_rf_active_recofit.md` and `reports/loso_rf_exercise_recofit.md`; put the
   pooled tables in the Written Report Outline (docs/05 Phase 1).
2. Calibrate the rep counter per exercise (`eval repcount --min-distance-s 1.2`) and record
   the choice in DECISIONS.
3. PR 6: compare `--model cnn` and `--model cnn-int8` with the RF rows on the same folds.
