"""
End-to-end v2 pipeline, usable from the dashboard, notebooks or the command line.

    Video -> [1] scene cuts -> [2] frames -> [3] YOLO -> [4] Verification
          -> [5] Context -> [6] Severity -> [7] Routing -> run folder (JSON + CSV)

Every run is saved to outputs/runs/<timestamp>/ so it can be labelled and
evaluated later (see evaluation/).

CLI:
    python -m pipeline.run_pipeline --video path.mp4 --weather "Clear" --conf 0.4 --interval 5
"""
import csv
import json
import os
import time

from agents.verification_agent import verify_incidents
from agents.context_agent import context_agent
from agents.severity_agent import severity_agent
from agents.routing_agent import routing_agent, build_final_incident
from pipeline.frame_extractor import extract_frames
from pipeline.scene_segmenter import detect_scene_cuts, build_segments
from pipeline.yolo_inference import run_detection
from config import settings

STEPS = ["Scene cuts", "Frames", "Detection", "Verification", "Context", "Severity", "Routing"]

CSV_FIELDS = [
    "incident_id", "anomaly", "severity", "severity_score", "segment_id", "location",
    "evidence_frames", "detections", "avg_confidence", "weather", "weather_source",
    "risk_level", "context_conclusion", "department", "recommended_action",
]


def new_run_dir(base=None):
    base = base or settings.RUNS_DIR
    d = os.path.join(base, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(d, exist_ok=True)
    return d


def write_incidents_csv(final_incidents, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for inc in final_incidents:
            w.writerow(inc)


def run_pipeline(video_path, model, weather_condition="Unknown", conf_threshold=0.4,
                 frame_interval=5, detect_cuts=True, run_dir=None, recorded_at=None,
                 on_step=None, on_progress=None):
    """
    on_step(index, name, info_dict) is called after each step (for live UIs).
    on_progress(done, total) is called during detection.
    Returns a result dict (also saved as run.json in run_dir).
    """
    run_dir = run_dir or new_run_dir()
    frames_dir = os.path.join(run_dir, "frames")
    notify = on_step or (lambda *a: None)

    # 1 - scene cuts
    if detect_cuts:
        cuts, fps, total = detect_scene_cuts(video_path, sensitivity=settings.SCENE_CUT_SENSITIVITY)
    else:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        cuts = []
    segments = build_segments(cuts, fps, total)
    notify(0, STEPS[0], {"cuts": cuts, "segments": segments, "fps": fps})

    # 2 - frames
    frames, fps, fstats = extract_frames(video_path, frames_dir, frame_interval, cut_frames=cuts)
    notify(1, STEPS[1], fstats)

    # 3 - detection
    raw = run_detection(model, frames, conf_threshold=conf_threshold, progress_callback=on_progress)
    notify(2, STEPS[2], {"raw_detections": len(raw)})

    # 4 - verification
    verified, ignored = verify_incidents(
        raw, min_conf=conf_threshold, frame_interval=fstats["frame_interval"],
        max_missed_frames=settings.VERIFY_MAX_MISSED_FRAMES,
        min_streak=settings.VERIFY_MIN_STREAK, fps=fps,
    )
    notify(3, STEPS[3], {"verified": verified, "ignored": ignored})

    # 5-7 - context, severity, routing
    weather = {"condition": weather_condition,
               "source": "manual" if weather_condition != "Unknown" else "unknown"}
    ctxs, sevs, finals = [], [], []
    for inc in verified:
        ctxs.append(context_agent(inc, weather, repetition_count=1))
    notify(4, STEPS[4], {"context": ctxs})
    for inc, ctx in zip(verified, ctxs):
        sevs.append(severity_agent(inc, ctx, repetition_count=1))
    notify(5, STEPS[5], {"severity": sevs})
    for inc, ctx, sev in zip(verified, ctxs, sevs):
        finals.append(build_final_incident(inc, ctx, sev, routing_agent(inc, sev)))
    notify(6, STEPS[6], {"final": finals})

    result = {
        "meta": {
            "video": os.path.basename(video_path),
            "recorded_at": recorded_at,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fps": fps,
            "conf_threshold": conf_threshold,
            "frame_interval": fstats["frame_interval"],
            "weather": weather,
            "scene_cut_detection": detect_cuts,
            "pipeline_version": "v2",
        },
        "segments": segments,
        "frame_stats": fstats,
        "frames": frames,
        "raw_detections": raw,
        "verified_incidents": verified,
        "ignored_repaired": ignored,
        "final_incidents": finals,
        "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=list)
    write_incidents_csv(finals, os.path.join(run_dir, "incidents.csv"))
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--model", default=settings.MODEL_PATH)
    p.add_argument("--weather", default="Unknown")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--interval", type=int, default=settings.DEFAULT_FRAME_INTERVAL)
    p.add_argument("--no-cuts", action="store_true")
    a = p.parse_args()
    r = run_pipeline(a.video, a.model, a.weather, a.conf, a.interval, detect_cuts=not a.no_cuts,
                     on_step=lambda i, n, info: print(f"[{i + 1}/7] {n} done"))
    print(f"frames={r['frame_stats']['frames_saved']} raw={len(r['raw_detections'])} "
          f"verified={len(r['verified_incidents'])} ignored(repair)={len(r['ignored_repaired'])}")
    print("saved to", r["run_dir"])
