# Road Anomaly Detection — Multi-Agent Pipeline (v2)

Detects road damage in dashcam video with a fine-tuned **YOLO11s** (RDD2022) and turns
raw detections into verified, prioritised and routed maintenance incidents through
four rule-based agents.

> v2 is a copy of the original `road_anomaly_project` (v1), which is left untouched.
> See **Changelog** below for every difference.

## Pipeline

```
Dashcam video
  1. Scene cuts        pipeline/scene_segmenter.py   split joined clips into segments
  2. Frames            pipeline/frame_extractor.py   every Nth frame, duplicate frames skipped
  3. Detection         pipeline/yolo_inference.py    YOLO11s → class, confidence, box
  4. Verification      agents/verification_agent.py  merge repeated sightings → incidents
  5. Context           agents/context_agent.py       weather-aware interpretation + risk
  6. Severity          agents/severity_agent.py      score + CRITICAL/HIGH/MEDIUM/LOW
  7. Routing           agents/routing_agent.py       department + action deadline
  → dashboard (timeline, gallery, CSV/JSON) + outputs/runs/<timestamp>/
```

Orchestration: `pipeline/run_pipeline.py` (used by the dashboard, the CLI and the evaluation scripts).

Model classes: `D00_longitudinal_crack`, `D10_transverse_crack`, `D20_alligator_crack`, `D40_pothole`, `repair`.
Baseline: mAP50 0.506 overall (79 epochs), D40 0.655, repair 0.393.

## User inputs (dashboard sidebar)

| Input | Why | Effect on output |
|---|---|---|
| Dashcam video | source | — |
| Weather at recording time: Unknown / Clear / Cloudy / Raining / Recently rained | Context Agent needs the conditions *when the video was shot* | Raining or Recently rained → cracks & potholes get risk "Increased" (+25 severity). Unknown → no adjustment, stated in the report |
| Recording date (optional) | report metadata | shown in summary / JSON |
| Detection mode: High recall 0.25 / Balanced 0.40 / High precision 0.60 | plain-language confidence threshold | lower → more detections & more false alarms; higher → fewer, more certain |
| Frame interval 1–15 | speed vs. evidence | lower → more frames per defect → higher evidence score, slower |
| Detect scene cuts | joined clips of different roads | segments never merge across a cut |

Moved out of the UI: model path (`config/settings.py`), OpenWeatherMap key (not used in v2).
Not used in v2: latitude/longitude and speed (no GPS) — incidents are located by **segment + video time**.

## Agents

**Verification Agent** — detections are grouped per *(segment, class)*. A detection joins the
current group if it is at most `(max_missed_frames + 1) × frame_interval` raw frames after the
previous one (default: one missed sampled frame allowed). A group is a VERIFIED incident only
if seen in ≥ 2 **distinct** frames. `repair` groups are set aside as already-repaired sections.

**Context Agent** — rule table:

| Anomaly | Raining | Recently rained | Recurring (future) | Otherwise |
|---|---|---|---|---|
| Crack | seepage widens crack → Increased | trapped water keeps seeping → Increased | monitor → Increased | Normal |
| Pothole | depth hidden by water → Increased | may be water-filled → Increased | worsening → Increased | Normal |

The optional `context_agent_llm` asks a local Ollama model only to *rephrase* the rule's decision.

**Severity Agent** (unchanged from v1)

```
score = avg_confidence × 30 + min(evidence_frames, 10) × 2 + risk (0 | 25) + min(repetition, 5) × 8
≥70 CRITICAL · ≥50 HIGH · ≥30 MEDIUM · else LOW
```
Each factor answers a separate question (how sure, how much evidence, how risky the context,
how persistent). Caps stop one factor dominating. The weights are hand-set; see
`evaluation/severity_sensitivity.py` for how sensitive the categories are to them.
`repetition_count` is fixed at 1 until incidents are stored across runs (future work).

**Routing Agent** — damage classes → Road Maintenance; CRITICAL 24 h, HIGH 3–5 days,
MEDIUM 2 weeks, LOW periodic log.

## Setup

```bash
pip install -r requirements.txt
# model: models/best.pt (change MODEL_PATH in config/settings.py if elsewhere)
streamlit run dashboard/app.py
```

Command line:
```bash
python -m pipeline.run_pipeline --video "path/to/video.mp4" --weather Clear --conf 0.4 --interval 5
```

## Evaluation (for the review)

```bash
# 1. v1 vs v2 on the same raw detections (model runs once per interval)
python -m evaluation.compare_v1_v2 --video "path/to/video.mp4" --interval 5 10 15

# 2. Does verification remove false positives?  (after a dashboard or CLI run)
python -m evaluation.export_label_sheet outputs/runs/<run_folder>
#    open label_sheet.csv, look at label_crops/, fill is_real with 1/0
python -m evaluation.evaluate_pipeline outputs/runs/<run_folder>

# 3. Why 30/2/25/8?  ±20 % per weight → how many incidents change category
python -m evaluation.severity_sensitivity outputs/runs/<run_folder>

# unit tests (no model needed)
python -m tests.test_agents
```

## Project layout

```
agents/        verification, context, severity, routing
pipeline/      scene_segmenter, frame_extractor, yolo_inference, run_pipeline
               (gps_tagger.py, weather_lookup.py kept for a future GPS version; unused in v2)
dashboard/     app.py  (v2 dashboard)
legacy/        app_v1.py, app_modular_v1.py  (original dashboards, for reference)
evaluation/    compare_v1_v2, export_label_sheet, evaluate_pipeline, severity_sensitivity, legacy_v1
config/        settings.py (model path, presets), load_keys.py
tests/         test_agents.py, fake_model.py (stand-in model for testing without ultralytics)
notebooks/     v1 development notebooks (self-contained, unchanged)
outputs/runs/  one folder per run: frames/, run.json, incidents.csv, label files
```

## Changelog v1 → v2

**Fixes**
- Frame intervals above 10 always produced **zero incidents** (fixed 10-frame gap). The gap now scales with the interval.
- A different class in the same frame (e.g. crack + pothole) **broke the streak** of the other class. Grouping is now per class.
- `evidence_frames` counted **boxes**, so two boxes in one frame counted as two frames. It now counts distinct frames.
- `repair` (already-repaired) detections appeared as incidents on the map. They are now listed separately.
- Duplicate frames from frame-rate conversion (184 of 922 in the test video) could be counted twice at interval 1. They are skipped.
- Unknown weather was silently treated as dry. It is now stated in the conclusion.
- Removed the Context Agent's "water" branch (the model has no water class).

**New**
- Scene-cut detection: joined clips become separate segments, and incidents never merge across a cut.
- Weather is entered for the *recording* time, with a new "Recently rained (wet road)" condition. `weather_source` is kept on every incident.
- Detection presets instead of a raw confidence slider. Model path moved to config.
- Severity score breakdown per factor, shown on every incident card.
- Dashboard: detection funnel, segment table, incident timeline (replaces the GPS map), CSV/JSON export, and every run saved to `outputs/runs/`.
- Evaluation scripts and unit tests.

**Removed from the UI:** latitude, longitude, assumed speed, OpenWeatherMap API key, model path.

## Known limitations
- No GPS: location is segment + timestamp only. Recurrence (`repetition_count`) needs locations across runs → future work.
- Weather is user-entered, not measured.
- Severity weights are hand-set (sensitivity analysis provided, not learned).
- Scene-cut detection finds hard cuts. Slow cross-fades may not be split.
- The model baseline is weak on `repair` and thin cracks. Model improvement is planned for Review 3.
