"""
Step 1 of pipeline evaluation: make a label sheet from a saved run.

    python -m evaluation.export_label_sheet outputs/runs/<run_folder>

Creates in the run folder:
  label_crops/det_00012.jpg   - each detection with its box drawn (plus context)
  label_sheet.csv             - one row per raw detection

Open label_sheet.csv in Excel, look at each crop, and fill the `is_real`
column with 1 (real defect of that class) or 0 (false detection / wrong class).
Then run evaluation/evaluate_pipeline.py.
"""
import csv
import json
import os
import sys

import cv2


def main(run_dir):
    with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as fh:
        run = json.load(fh)

    det_to_incident = {}
    for inc in run["verified_incidents"] + run["ignored_repaired"]:
        for d in inc.get("member_det_ids", []):
            det_to_incident[d] = inc["incident_id"]

    crops = os.path.join(run_dir, "label_crops")
    os.makedirs(crops, exist_ok=True)
    rows = []
    for det in run["raw_detections"]:
        img = cv2.imread(det["frame_path"])
        crop_name = f"det_{det['det_id']:05d}.jpg"
        if img is not None:
            h, w = img.shape[:2]
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            pad = 120
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
            cv2.imwrite(os.path.join(crops, crop_name), img[cy1:cy2, cx1:cx2])
        rows.append({
            "det_id": det["det_id"],
            "frame": det["frame"],
            "time_sec": det.get("timestamp_sec"),
            "segment_id": det.get("segment_id", 1),
            "class": det["class"],
            "conf": det["conf"],
            "in_incident": det_to_incident.get(det["det_id"], ""),
            "crop": os.path.join("label_crops", crop_name),
            "is_real": "",
            "notes": "",
        })

    out = os.path.join(run_dir, "label_sheet.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["det_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} detections -> {out}")
    print("Fill `is_real` with 1 or 0, then run: python -m evaluation.evaluate_pipeline", run_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
