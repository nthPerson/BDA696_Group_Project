# How to: the rules engine, rules.yaml, and the rules / gating evaluations

**Owner:** Rochelle Reyes (rules) · Christian Byars (gating benchmark) · **Built by:** Claude Code with Robert on 2026-09-24 · **PR:** #7
**Checkpoint:** 6 (`docs/00-START-HERE.md`) · **State:** works on real data (MM-Fit 3-D pose); thresholds are design defaults that the report shows need calibration

## 1. Run it (verified 2026-09-24 on Linux/WSL2)

Validate the packaged rules on every MM-Fit set of the four exercises (44 s) and write a
percentile-calibrated copy for this pose source:

```bash
uv run formcoach eval rules --calibrate-out src/formcoach/rules/rules.mmfit-pose3d.yaml
```
```
reps per exercise: {'curl': 586, 'press': 540, 'raise': 557, 'squat': 637}
wrote /home/…/reports/rules_validation.md and src/formcoach/rules/rules.mmfit-pose3d.yaml
```

Gating benchmark (always-on vs energy vs the RF laptop gate) on the replay fixture and any
`data/team/*/*` session with a `pose.parquet`; the laptop gate needs the RF model first:

```bash
uv run formcoach train gate --model rf          # ~3 min -> models/gate_rf.joblib
uv run formcoach eval gating
```
```
        session      gate  frames_total  frames_processed  frames_processed_pct  gated_s  imu_s  wall_s  cpu_pct  reps  faults  reps_match  faults_match
mmfit_w00_curls always_on           898               897                  99.9    29.98  29.98   2.189     85.8    10      25        True          True
mmfit_w00_curls    energy           898               744                  82.9    24.87  29.98   0.675     92.3    10      25        True          True
mmfit_w00_curls    laptop           898               711                  79.2    23.77  29.98   6.983     85.6     9      22       False         False
```

Replay with the rules attached (faults print in red):

```bash
uv run formcoach demo --source replay --headless --rules src/formcoach/rules/rules.mmfit-pose3d.yaml
```

## 2. Where things live

| Path | What it is |
|---|---|
| `src/formcoach/rules/rules.yaml` | docs/02 §7 catalogue: 16 rules with `code, severity, message, when: [{metric, op, threshold}]`, plus gate + rep-segmentation constants |
| `src/formcoach/rules/rules.mmfit-pose3d.yaml` | generated copy with thresholds at the 5th/95th percentile of MM-Fit correct-form reps |
| `src/formcoach/rules/engine.py` | `RuleEngine.from_yaml(path)`, `.evaluate(rep) -> list[Fault]`, `.rules_for(exercise)`, `.rep_config(exercise)` |
| `src/formcoach/app/pipeline.py` | `metrics_from_angles(res, duration_s)` — the metric keys rules refer to; `run_replay(..., rules=engine)` |
| `src/formcoach/eval/rules_eval.py` | `evaluate(raw_root, out, calibrate_out=...)`, `PERTURBATIONS`, `write_calibrated` |
| `src/formcoach/eval/gating.py` | `evaluate(sessions, gates, out)` → `reports/gating.md` |
| `src/formcoach/models/train.py` | `train_rf_gate(...)` → `models/gate_rf.joblib` (gitignored) for `--gate laptop` |
| `reports/rules_validation.md`, `reports/gating.md` | the generated reports (committed) |
| `tests/test_rules.py`, `tests/test_eval_rules_gating.py` | 27 tests: one perturbation per rule, correct reps pass, pipeline integration, report sections |

## 3. How it works

A rep arrives as `RepMetrics`: 17 angle series resampled to 30 steps plus scalar metrics
(`elbow_min`, `elbow_max`, `upper_arm_trunk_drift`, `elbow_drift_max`, `trunk_incl_max`,
`abd_peak`, `elbow_at_abd_peak`, `knee_min`, `trunk_incl_at_bottom`, `knee_track_at_bottom`,
`hip_knee_height_at_bottom`, `duration_s`, asymmetries). A rule fires when any of its `when`
conditions holds; missing/NaN metrics never fire. `eval rules` treats MM-Fit reps as correct
form: the pass rate is 1 − false-alarm rate, and the perturbation table applies synthetic faults
to the same angle series (ROM × 0.7, +25° trunk lean, speed × 2, elbow shifted 0.2 torso, …)
and reports how often the intended rule fires. With the design defaults, ROM rules fire on
almost every MM-Fit rep (lifted 3-D pose reads curls as 95–140° and presses with elbow max
≈ 108°), so absolute thresholds must be calibrated per pose source (ADR-0022) — that is what
the `--calibrate-out` file is.

## 4. Tests

`uv run pytest tests/test_rules.py tests/test_eval_rules_gating.py`. `test_rules.py` has one
case per rule: a correct synthetic rep raises nothing and each perturbation raises exactly its
code. Adding a rule = one YAML entry + one `PERTURBATIONS` row in the test. The report test
runs on the synthetic MM-Fit fixture (curl + squat sets).

## 5. Could not verify / open questions

- No MediaPipe run yet: pass rates and percentiles are for MM-Fit's lifted 3-D pose. Rochelle
  should rerun `eval rules` on MediaPipe world landmarks from one MM-Fit video and compare.
- Press segmentation is the weakest (540 of 598 reps, 5 % of sets exact): the press has a
  small elbow range in this pose source; the adaptive segmenter's `min_range_deg` may need to
  be per exercise.
- The RF laptop gate loses a rep at a gate boundary (9 vs 10 on the fixture): the 6 s
  close-delay may need to grow for slow exercises.
- CPU % is process CPU during a fastest-speed replay, not a live-camera measurement.

## 6. Next steps for Rochelle / Christian

1. Rochelle: decide per rule whether the calibrated threshold or the design value ships in
   `rules.yaml` v1 (DECISIONS entry), then rerun `eval rules`.
2. Christian: after PR 6, add `--gate device` rows to `eval gating` from recorded sessions
   with `flags.gate_state` (needs a board).
