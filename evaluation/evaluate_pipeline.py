"""
Step 2 of pipeline evaluation: measure what the Verification Agent actually does.

    python -m evaluation.evaluate_pipeline outputs/runs/<run_folder>

Needs label_sheet.csv with `is_real` filled (1 / 0).

Reports
  * precision of raw YOLO detections            = real / all raw detections
  * precision of detections kept in incidents   = real / detections inside verified incidents
  * false positives removed by verification     = false dets NOT in any incident / all false dets
  * real detections lost by verification        = real dets NOT in any incident / all real dets
  * incident-level precision                    = incidents whose detections are mostly real
"""
import csv
import os
import sys
from collections import defaultdict


def pct(a, b):
    return f"{(100.0 * a / b):.1f}%" if b else "n/a"


def main(run_dir):
    path = os.path.join(run_dir, "label_sheet.csv")
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["is_real"].strip() in ("0", "1")]
    if not rows:
        sys.exit("No labelled rows found - fill the is_real column with 1 or 0 first.")

    real = [r for r in rows if r["is_real"].strip() == "1"]
    fake = [r for r in rows if r["is_real"].strip() == "0"]
    kept = [r for r in rows if r["in_incident"] and "REPAIR" not in r["in_incident"]]
    kept_real = [r for r in kept if r["is_real"].strip() == "1"]
    fake_removed = [r for r in fake if not r["in_incident"]]
    real_lost = [r for r in real if not r["in_incident"]]

    by_inc = defaultdict(list)
    for r in kept:
        by_inc[r["in_incident"]].append(int(r["is_real"]))
    good_inc = sum(1 for v in by_inc.values() if sum(v) / len(v) >= 0.5)

    lines = [
        f"Labelled detections           : {len(rows)}  (real {len(real)}, false {len(fake)})",
        f"Raw detection precision       : {pct(len(real), len(rows))}",
        f"Precision after verification  : {pct(len(kept_real), len(kept))}  ({len(kept)} detections kept)",
        f"False positives removed       : {pct(len(fake_removed), len(fake))}  ({len(fake_removed)}/{len(fake)})",
        f"Real detections lost          : {pct(len(real_lost), len(real))}  ({len(real_lost)}/{len(real)})",
        f"Incident-level precision      : {pct(good_inc, len(by_inc))}  ({good_inc}/{len(by_inc)} incidents mostly real)",
    ]
    per_class = defaultdict(lambda: [0, 0])
    for r in rows:
        per_class[r["class"]][0] += int(r["is_real"])
        per_class[r["class"]][1] += 1
    lines.append("Raw precision per class       :")
    for c, (ok, n) in sorted(per_class.items()):
        lines.append(f"    {c:<26} {pct(ok, n)}  ({ok}/{n})")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(run_dir, "evaluation_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
