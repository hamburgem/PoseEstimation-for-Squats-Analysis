# Squat Analysis — Real-Time Pose Estimation

A desktop application that uses **MediaPipe Pose** and **OpenCV** to analyse squat form in real time from a webcam. It estimates body landmarks, computes biomechanical angles, runs a **finite-state machine** for rep counting, and surfaces **view-aware feedback** (front / side / back) so users can train with actionable cues.

---

## Why this project (interview pitch)

**Problem:** Manual coaching does not scale; subjective feedback is inconsistent. Computer vision can give immediate, repeatable cues—if the pipeline is robust to camera angle, occlusion, and noisy 2D projections.

**What I built:** An end-to-end pipeline from **video capture → pose landmarks → angle geometry → state-based rep logic → HUD**, with explicit handling of **front vs side vs back** views (including a dedicated depth signal for frontal squats where knee flexion is poorly visible in 2D).

**Skills demonstrated:** Python, OpenCV, MediaPipe Tasks API, geometry for pose metrics, state machines, UX for real-time overlays, threshold tuning for noisy vision data.

---

## Demo videos

| Scenario | Video |
|---|---|
| **Correct rep counting** | [Watch demo](https://youtu.be/1AifvK2JHu0) |
| **Incorrect rep detection** | [Watch demo](https://youtu.be/FnGWWFHPrtk) |

---

## Features

- **Live pose overlay** — skeleton and keypoints on the camera feed (profile view can show a single-side chain to reduce clutter).
- **Modes** — Auto, Front, Side, Back; optional skill level (e.g. beginner / pro thresholds).
- **Rep counting** — State machine (`s1` / `s2` / `s3`) driven by depth metrics; separate logic where frontal depth uses **hip-drop** instead of unreliable 2D knee angle alone.
- **Feedback** — View-specific rules (e.g. symmetry and valgus cues in front; depth, torso, knee travel in side) with short on-screen messages.
- **Set control** — Start / end set (keyboard and UI), rep tallies for correct vs incorrect.

---

## Tech stack

| Layer | Technology |
|--------|------------|
| Language | Python 3 |
| Vision | OpenCV (`opencv-python`) |
| Pose | MediaPipe Pose Landmarker (`.task` model bundled in repo) |
| Math | NumPy (angles, vectors) |

---

## Project layout (high level)

| File | Role |
|------|------|
| `main.py` | Webcam loop, detector, angle pipeline, state machine, HUD, input handling |
| `pose_detector.py` | MediaPipe inference + landmark drawing (full vs single-side profile) |
| `angles.py` | Joint angles, view scoring, depth metrics (`depth_metric`, `hip_drop_metric`, etc.) |
| `state_machine.py` | Rep FSM, thresholds, per-view rules, feedback timing |
| `hud.py` | Overlays: counters, stats, mode panel, feedback cards |
| `pose_landmarker_lite.task` | Bundled pose model (required at runtime) |

---

## Setup

**Requirements:** Python 3.10+ recommended, webcam, the MediaPipe model file present next to `pose_detector.py`.

```bash
cd PoseEstimation-for-Squats-Analysis
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python main.py
```

**Controls (typical):** `Q` quit · `A`/`F`/`S`/`B` mode · `Space` start/end set · skill keys as implemented in `main.py`.

---

## Limitations & honesty (good to mention in interviews)

- **2D monocular** pose: depth and occlusion are fundamentally ambiguous; thresholds are tuned heuristically, not clinical-grade.
- **Front camera:** knee flexion along the optical axis is weak in 2D; the project uses **hip vertical motion** (among other signals) to approximate squat depth.
- **Performance** depends on lighting, distance, and `pose_landmarker_lite` vs heavier models.

---

## License / attribution

Add your license choice here if you open-source the repo. MediaPipe and model use are subject to [Google’s MediaPipe terms](https://developers.google.com/mediapipe) and bundled asset licenses.

---

## Author

**Your name** — _[LinkedIn / portfolio / email — optional]_

_Last updated: [DATE]_ — replace with the date you hand this in.
