# 05 — Roadmap, milestones, roles, definition of done

Working weeks run late September → early December 2026. Course deliverable due dates are **TBD per syllabus** — Robert fills in the table below; until then plan by phase windows. The final presentation is expected on December 2 or 9.

## 1. Course deliverables

| Deliverable | Due (fill in) | Produced from |
|---|---|---|
| Pitch Presentation | done (late Sep) | `docs/pitch/` (slides + PDF, archived) |
| Written Report Outline | ____ | `reports/` baseline tables + `docs/report/outline.md` |
| First Draft | ____ | `docs/report/draft.md` + regenerated `reports/` |
| Final Presentation Slides | ____ | `docs/slides/` |
| Final Written Report | ____ | `docs/report/final.md` (+ export to .docx/.pdf) |
| Final Presentation | Dec 2 or 9 | live demo + slides |

## 2. Phases

### Phase 1 — Data, baselines, pose PoC (Sep 23 – Oct 6)

- [ ] Repo skeleton, CI, README quickstart (Checkpoint 0)
- [ ] MM-Fit, RecoFit, RecGym fetch + loaders + profile report (Checkpoint 1)
- [ ] Signal pipeline, windows/features, LOSO utility, RF baseline, peak-count baseline (Checkpoint 2)
- [ ] Pose extraction, angle module, rep segmentation, replay demo (Checkpoint 3)
- [ ] Parts ordered; case v1 modeled and test-printed (Robert)
- [ ] PlatformIO project compiles in CI (no hardware needed)
- [ ] `docs/report/outline.md` drafted from the profile + baseline reports → **Written Report Outline**

### Phase 2 — Build, firmware, gate on device, rules v1 (Oct 7 – Oct 27)

- [ ] Build session: 5 + 1 units assembled, flashed, registered in `docs/devices.md`; wiring guide with photos
- [ ] Firmware v0 (serial CSV) and v1 (BLE protocol); `record` and `demo --source ble` work on Linux/macOS/Windows
- [ ] Keras 1D-CNN gate trained on RecoFit + MM-Fit; LOSO vs RF; int8 export; deployed; on-device latency/RAM/flash measured (Checkpoint 5)
- [ ] `rules.yaml` v1 for four exercises; rules unit tests; `eval rules` on MM-Fit
- [ ] End-to-end demo: wearable → gate → pose → rules → overlay + LED; 30-second demo video
- [ ] `docs/report/draft.md` → **First Draft**

### Phase 3 — Validation, ablations, benchmark, freeze (Oct 28 – Nov 17)

- [ ] Team validation recordings (~1 h) per `docs/04-datasets.md` §6; `eval transfer`
- [ ] `eval gating` (always-on / energy / laptop / device) on recorded sessions
- [ ] `eval pose-ablation` (Lite / Full / Heavy; YOLO11n-pose optional)
- [ ] `eval latency`, `eval device`, battery test
- [ ] **Go/no-go on the vision path (~Nov 3)**: if rep segmentation agreement or rule validity is unacceptable, switch to the fallback in `docs/02-system-design.md` §9
- [ ] Feature freeze ~Nov 10; `make eval` regenerates every table and figure
- [ ] Streamlit session dashboard v1; Dockerfile for the replay path
- [ ] `docs/slides/` → **Final Presentation Slides**

### Phase 4 — Report, video, presentation (Nov 18 – Dec 9)

- [ ] Final report with all results produced by scripts; error analysis; limitations; reproducibility appendix
- [ ] live-demo failure rehearsal (BLE drop → serial; camera missing → replay)
- [ ] README followed by a teammate from a fresh clone (recorded as a test)
- [ ] Anonymized team validation set archived with manifest
- [ ] **Final Written Report** and **Final Presentation**

## 3. Roles (provisional; build so anyone can run anything)

| Member | Owns | First tasks |
|---|---|---|
| Robert Ashe (lead) | Hardware, firmware, case, architecture, GitHub | Order parts, case v1, firmware v0, build session |
| Ruby Rios Ramirez | Data engineering | Loaders, profile report, Parquet store, manifest |
| Rochelle Reyes | Vision & rules | Pose extraction, angles, rep segmentation, `rules.yaml` |
| Christian Byars | Embedded ML & evaluation | RF baseline, CNN gate, quantization, LOSO + gating harness |
| Bryce Hall | Application, docs & demo | Overlay app, dashboard, README, report assembly, slides, video |

When you (Claude Code) do a teammate's task because they are unavailable, write it so they can take it over: docstrings, a `docs/howto/<topic>.md`, and a STATUS entry naming what they should read.

## 4. Stretch goals (off-limits until feature freeze)

1. Coaching sentences from fault codes via a local LLM (Ollama) or an API key Robert provides; suggest an easier regression exercise.
2. Learned per-rep fault classifier (only if Fitness-AQA access is granted).
3. Personalization experiment: last-layer fine-tuning on one subject's reps.
4. Physical-therapy framing (shoulder range-of-motion tracking).
5. Second camera / side-view robustness.

## 5. Definition of done

- `git clone && make setup && make demo` runs the replay pipeline on public data on Linux, macOS and Windows, and a teammate has confirmed it.
- Five wearables run the same firmware version, stream over BLE, and are registered in `docs/devices.md`; wiring guide and printable case files are in the repo.
- `make eval` regenerates every table and figure used in the report: LOSO results for baseline/RF/CNN/int8, cross-dataset test, rep-count MAE, rule validation, gated-vs-always-on efficiency, pose ablation, latency, on-device metrics, sensor transfer.
- `docs/STATUS.md` and `docs/DECISIONS.md` tell the story of how the project got here.
- Final report, slides and demo video are in `docs/`, and every number in them traces to a script.
