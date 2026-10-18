"""
Run the model ONCE on a video, then feed the same raw detections to the v1 and
v2 verification logic and print a side-by-side table for the Review 2 slides.

Run on the laptop (needs ultralytics), from the project root:

    python -m evaluation.compare_v1_v2 --video "data/raw_videos/Video_Project (1).mp4"

Options: --conf 0.4  --interval 5 10 15  --weather Clear
Writes outputs/runs/compare_<timestamp>/comparison.csv and the v2 run.json.
"""
import argparse
import csv
import copy
import os
import time

from agents.verification_agent import verify_incidents
from agents.context_agent import context_agent
from agents.severity_agent import severity_agent
from config import settings
from evaluation.legacy_v1 import simulate_gps_from_start, verify_incidents_v1
from pipeline.frame_extractor import extract_frames
from pipeline.scene_segmenter import detect_scene_cuts
from pipeline.yolo_inference import run_detection


def severity_counts(incidents, weather):
    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for inc in incidents:
        ctx = context_agent(inc, weather)
        counts[severity_agent(inc, ctx)["severity"]] += 1
    return counts


def compare(video, model, conf, interval, weather_condition, out_dir):
    frames_dir = os.path.join(out_dir, f"frames_i{interval}")
    cuts, fps, _ = detect_scene_cuts(video, sensitivity=settings.SCENE_CUT_SENSITIVITY)
    frames, fps, fstats = extract_frames(video, frames_dir, interval, cut_frames=cuts, skip_duplicates=False)
    raw = run_detection(model, frames, conf_threshold=conf)

    # v1: GPS simulation + original verification
    frames_v1 = simulate_gps_from_start(copy.deepcopy(frames), fps=fps, frame_interval=interval)
    gps_by_frame = {f["frame_number"]: f["gps"] for f in frames_v1}
    raw_v1 = [dict(d, gps=gps_by_frame[d["frame"]]) for d in raw]
    v1 = verify_incidents_v1(raw_v1, min_conf=conf)
    v1_repair = [i for i in v1 if i["anomaly"] == "repair"]

    # v2
    v2, v2_ignored = verify_incidents(raw, min_conf=conf, frame_interval=interval, fps=fps,
                                      max_missed_frames=settings.VERIFY_MAX_MISSED_FRAMES,
                                      min_streak=settings.VERIFY_MIN_STREAK)
    weather = {"condition": weather_condition, "source": "manual"}
    in_v1 = {d for i in v1 for d in i["member_det_ids"]}
    in_v2 = {d for i in v2 + v2_ignored for d in i["member_det_ids"]}

    return {
        "interval": interval,
        "scene_cuts": len(cuts),
        "frames": len(frames),
        "raw_detections": len(raw),
        "v1_incidents": len(v1),
        "v1_repair_shown_as_incident": len(v1_repair),
        "v2_incidents": len(v2),
        "v2_repair_ignored": len(v2_ignored),
        "raw_dets_used_v1": len(in_v1),
        "raw_dets_used_v2": len(in_v2),
        "v1_severity": severity_counts([i for i in v1 if i["anomaly"] != "repair"], weather),
        "v2_severity": severity_counts(v2, weather),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--model", default=settings.MODEL_PATH)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--interval", type=int, nargs="+", default=[5, 10, 15])
    p.add_argument("--weather", default="Clear")
    a = p.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    out_dir = os.path.join(settings.RUNS_DIR, "compare_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for iv in a.interval:
        print(f"interval {iv} ...")
        rows.append(compare(a.video, model, a.conf, iv, a.weather, out_dir))

    print("\n{:>8} {:>6} {:>5} {:>8} {:>8} {:>8} {:>10}".format(
        "interval", "frames", "raw", "v1 inc", "v2 inc", "v1 rep*", "v2 ignored"))
    for r in rows:
        print("{interval:>8} {frames:>6} {raw_detections:>5} {v1_incidents:>8} {v2_incidents:>8} "
              "{v1_repair_shown_as_incident:>8} {v2_repair_ignored:>10}".format(**r))
        print(f"          severity v1 {r['v1_severity']}  |  v2 {r['v2_severity']}")
    print("* v1 rep = 'repair' groups that v1 counted as incidents")

    path = os.path.join(out_dir, "comparison.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nsaved", path)


if __name__ == "__main__":
    main()
