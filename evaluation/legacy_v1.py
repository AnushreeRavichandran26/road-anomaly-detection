"""
Exact copy of the v1 verification logic and GPS simulation, used ONLY by
compare_v1_v2.py so both versions can be run on the same raw detections.
Do not edit - it is the reference baseline.
"""
from math import radians, sin, cos, sqrt, atan2


def simulate_gps_from_start(frame_records, start_lat=13.0827, start_lon=80.2707,
                            assumed_speed_kmh=30, fps=25.0, frame_interval=5):
    speed_m_s = assumed_speed_kmh * 1000 / 3600
    seconds_per_extracted_frame = frame_interval / fps if fps > 0 else 0
    meters_per_frame_step = speed_m_s * seconds_per_extracted_frame
    for idx, rec in enumerate(frame_records):
        delta_lat = (meters_per_frame_step * idx) / 111320.0
        rec["gps"] = (round(start_lat + delta_lat, 6), round(start_lon, 6))
    return frame_records


def gps_distance_meters(gps1, gps2):
    R = 6371000
    lat1, lon1 = radians(gps1[0]), radians(gps1[1])
    lat2, lon2 = radians(gps2[0]), radians(gps2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def build_incident(group):
    avg_conf = sum(d["conf"] for d in group) / len(group)
    return {
        "incident_id": f"RA-{group[0]['frame']:04d}",
        "anomaly": group[0]["class"],
        "status": "VERIFIED",
        "evidence_frames": len(group),
        "avg_confidence": round(avg_conf, 3),
        "gps": group[len(group) // 2]["gps"],
        "frame_range": (group[0]["frame"], group[-1]["frame"]),
        "member_frames": [d["frame"] for d in group],
        "member_det_ids": [d["det_id"] for d in group if "det_id" in d],
    }


def verify_incidents_v1(detections, min_conf=0.4, max_frame_gap=10, max_gps_dist=25, min_streak=2):
    detections = sorted(detections, key=lambda d: d["frame"])
    incidents, current_group = [], []
    for det in detections:
        if det["conf"] < min_conf:
            continue
        if not current_group:
            current_group = [det]
            continue
        last = current_group[-1]
        if (det["class"] == last["class"]
                and (det["frame"] - last["frame"]) <= max_frame_gap
                and gps_distance_meters(det["gps"], last["gps"]) <= max_gps_dist):
            current_group.append(det)
        else:
            if len(current_group) >= min_streak:
                incidents.append(build_incident(current_group))
            current_group = [det]
    if len(current_group) >= min_streak:
        incidents.append(build_incident(current_group))
    return incidents
