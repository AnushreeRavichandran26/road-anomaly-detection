"""
Verification Agent (v2)

Problem: one physical defect is seen in many consecutive frames as the vehicle
drives past it, and one-off false positives appear in single frames.

Rule (v2)
---------
Detections are grouped **per (segment, class)**. Inside a group, detections are
merged into one incident while consecutive frames are no more than
`max_missed_frames` sampled frames apart. A merged group becomes a VERIFIED
incident only if it was seen in at least `min_streak` *distinct* frames.

v1 -> v2 changes
* per-class grouping: a pothole in the same frame no longer breaks a crack streak
* grouping resets at scene cuts (segment_id): clips of different roads never merge
* frame gap scales with frame_interval, so any interval works
  (v1 used a fixed 10-frame gap -> intervals > 10 always gave zero incidents)
* evidence = number of DISTINCT frames (v1 counted boxes; 2 boxes in 1 frame = 2)
* `repair` detections are returned separately as already-repaired sections
* GPS distance check removed (no GPS input in this version)
"""
from collections import defaultdict

IGNORED_CLASSES = ("repair",)


def _class_code(cls_name):
    return cls_name.split("_")[0].upper()


def build_incident(group, fps=None):
    """Summarise one group of raw detections (same segment + class)."""
    by_frame = defaultdict(list)
    for d in group:
        by_frame[d["frame"]].append(d)
    frames = sorted(by_frame)
    # per frame keep the most confident box, then average across frames
    best_per_frame = [max(by_frame[f], key=lambda d: d["conf"]) for f in frames]
    avg_conf = sum(d["conf"] for d in best_per_frame) / len(best_per_frame)

    t0 = best_per_frame[0].get("timestamp_sec")
    t1 = best_per_frame[-1].get("timestamp_sec")
    if t0 is None and fps:
        t0, t1 = round(frames[0] / fps, 2), round(frames[-1] / fps, 2)
    seg = group[0].get("segment_id", 1)

    return {
        "incident_id": f"RA-{frames[0]:05d}-{_class_code(group[0]['class'])}",
        "anomaly": group[0]["class"],
        "status": "VERIFIED",
        "segment_id": seg,
        "evidence_frames": len(frames),
        "detections": len(group),
        "avg_confidence": round(avg_conf, 3),
        "max_confidence": round(max(d["conf"] for d in group), 3),
        "frame_range": (frames[0], frames[-1]),
        "time_range_sec": (t0, t1),
        "location_label": f"Segment {seg} · {t0:.1f}–{t1:.1f} s" if t0 is not None else f"Segment {seg}",
        "member_frames": frames,
        "member_det_ids": [d["det_id"] for d in group if "det_id" in d],
    }


def verify_incidents(detections, min_conf=0.4, frame_interval=5, max_missed_frames=1,
                     min_streak=2, fps=None, ignored_classes=IGNORED_CLASSES):
    """
    Returns (incidents, ignored).

    min_conf          : detections below this confidence are dropped
    frame_interval    : the sampling interval used by the frame extractor
    max_missed_frames : how many sampled frames in a row a defect may be missed
                        (occlusion / low confidence) before the streak breaks
    min_streak        : minimum DISTINCT frames needed to confirm an incident
    """
    max_frame_gap = (max_missed_frames + 1) * max(1, int(frame_interval))

    buckets = defaultdict(list)
    for det in detections:
        if det["conf"] < min_conf:
            continue
        buckets[(det.get("segment_id", 1), det["class"])].append(det)

    incidents, ignored = [], []
    for (seg, cls), dets in buckets.items():
        dets.sort(key=lambda d: d["frame"])
        group = [dets[0]]
        groups = []
        for det in dets[1:]:
            if det["frame"] - group[-1]["frame"] <= max_frame_gap:
                group.append(det)
            else:
                groups.append(group)
                group = [det]
        groups.append(group)

        for g in groups:
            if len({d["frame"] for d in g}) < min_streak:
                continue
            inc = build_incident(g, fps)
            if cls in ignored_classes:
                inc["status"] = "IGNORED"
                inc["reason"] = "Already-repaired section"
                ignored.append(inc)
            else:
                incidents.append(inc)

    incidents.sort(key=lambda i: (i["frame_range"][0], i["anomaly"]))
    ignored.sort(key=lambda i: i["frame_range"][0])
    return incidents, ignored


if __name__ == "__main__":
    sample = [
        {"det_id": 0, "frame": 10, "segment_id": 1, "class": "D40_pothole", "conf": 0.75, "timestamp_sec": 0.33},
        {"det_id": 1, "frame": 10, "segment_id": 1, "class": "D00_longitudinal_crack", "conf": 0.6, "timestamp_sec": 0.33},
        {"det_id": 2, "frame": 15, "segment_id": 1, "class": "D40_pothole", "conf": 0.82, "timestamp_sec": 0.5},
    ]
    inc, ign = verify_incidents(sample)
    print("Verification agent smoke test passed:", len(inc) == 1 and inc[0]["evidence_frames"] == 2)
