# 01 — Project brief

## Course context

- **Course:** BDA 696 (Big Data Analytics program, San Diego State University), Fall 2026. Class meets Wednesdays 4:00–6:40 pm. Semester ends December 11, 2026.
- **Assignment:** Final Group Project. Graded artifacts, in order: Pitch Presentation (done, late September) → **Written Report Outline** → **First Draft** → **Final Presentation Slides** → **Final Written Report** → **Final Presentation**. Exact due dates come from the syllabus; Robert will put them in `docs/05-roadmap.md` when confirmed. Plan for the final presentation on **December 2 or 9**.
- **What graders look for** (from the assignment): a focused objective; sufficiently large and complex data (tabular ≥ 25 k observations × 15 features, image ≥ 1 k images, or gridded > 5 GB — we exceed all of these); a scalable, documented, reproducible Python processing workflow; an appropriate model or AI system with **at least two compared methods plus a baseline**; a sound evaluation strategy with the right metrics; a realistic roadmap; clear documentation and GitHub; group participation. Python is the primary language.
- **Instructor expectation set in class:** real applications, not "a couple two-file Python things with messed-up comments". Code quality, structure and documentation count.

## Team

| Member | Provisional role | Notes |
|---|---|---|
| Robert Ashe (lead) | Hardware, firmware, case, GitHub, architecture | Your operator. Owns soldering gear, two 3D printers, Fusion 360. |
| Ruby Rios Ramirez | Data engineering | Loaders, profiling, cleaning, Parquet store, manifest |
| Rochelle Reyes | Vision & rules | MediaPipe pipeline, angles, rep segmentation, form rules |
| Christian Byars | Embedded ML & evaluation | Gate model, quantization, LOSO metrics, gating benchmark |
| Bryce Hall | Application, docs & demo | Live app, dashboard, README, report assembly, slides, video |

Roles are provisional. Build so that any teammate can run any part. Everyone shares EDA, writing and slides. Expect uneven availability; design tasks that one person can finish in a week without supervision, and make the repo self-explanatory.

## The project in one paragraph

**FormCoach** is a two-part system. A wrist wearable the team assembles (Seeed XIAO ESP32-S3, Bosch BMI160 6-axis IMU, 300 mAh LiPo, slide switch, one push button, 3D-printed case with Apple Watch band lugs) runs a tiny quantized neural network that reports "exercise in progress" over Bluetooth. A Python application on a laptop uses that signal as a *gate*: only while the gate is open does it run MediaPipe pose estimation on the laptop's webcam, compute joint angles, segment repetitions, and apply per-exercise biomechanical rules to flag form faults (elbow swing on a curl, shallow squat, incomplete press lockout, over-raised lateral raise, too-fast tempo). Feedback is drawn on screen and echoed as an LED code on the wearable. The engineering claim under test: **sensor-gated vision gives the same per-rep feedback as always-on vision while processing a fraction of the frames on an ordinary laptop CPU.** Scope is frozen to four dumbbell exercises: bicep curl, overhead press, lateral raise, squat.

## Why it is framed this way

- **Public data is primary.** The team cannot run a data-collection campaign. MM-Fit and RecoFit train and evaluate everything learned; a small team validation set (~1 hour) only checks that models transfer to our sensor and that the live pipeline works.
- **Rules, not a learned fault classifier.** No public dataset labels form faults for our exercises, so form feedback is rule-based on top of a pretrained pose model, calibrated on MM-Fit pose sequences and tested with synthetic perturbations. A learned classifier is a stretch goal only.
- **The learned component is the gate / exercise recognizer.** It is where the "at least two methods plus baseline" requirement is met: motion-energy threshold (baseline) → random forest on hand-crafted features → 1D-CNN (float and int8). The pose model is compared across variants (MediaPipe Lite/Full/Heavy, optionally YOLO11n-pose), and the architecture is compared gated vs always-on.
- **Hardware is deliberately minimal**: five electronic parts, nine solder joints, ~$16 per unit. Teammates are not electronics people; one build session led by Robert must produce five working units.

## Constraints and assumptions

- Laptops without discrete GPUs; everything must run on CPU. Mixed Linux / macOS / Windows.
- Budget: parts already ~$155 for the team; no further purchases without asking.
- No cloud services are required; nothing in the core path may depend on a paid API. (An LLM "coaching sentence" extension may use Ollama locally or an API key that Robert provides; it is a stretch goal.)
- Privacy: team video never leaves teammates' machines except pose Parquet files; subjects are coded S1–S5.
- Time: roughly ten working weeks from late September, with a report and presentation at the end. Robert has other heavy commitments; the repo must let the team progress without him for a week at a time.

## Success criteria (what the final report must be able to say)

1. Exercise-recognition and gate accuracy on **unseen subjects** (LOSO on RecoFit; cross-dataset check on MM-Fit), for baseline, RF and CNN, float and int8.
2. Rep-count MAE against MM-Fit labels for IMU-only, pose-only and fused segmentation, versus peak-detection baseline.
3. Gated vs always-on: frames processed, CPU %, wall time (and power where measurable) on identical recordings, with identical feedback outputs.
4. Rule validation: pass rate on correct-form MM-Fit reps, detection under synthetic perturbations, agreement on team validation reps.
5. On-device: inference latency, RAM/flash, battery life per workout; end-to-end rep-end → feedback latency < 500 ms.
6. Five working wearables, a wiring guide, printable case files, and a README that a teammate followed successfully.
