"""
Unit tests for the v2 agents - no model, no video needed.

    python -m tests.test_agents
"""
from agents.verification_agent import verify_incidents
from agents.context_agent import context_agent
from agents.severity_agent import severity_agent
from agents.routing_agent import routing_agent, build_final_incident


def det(i, frame, cls, conf, seg=1):
    return {"det_id": i, "frame": frame, "segment_id": seg, "class": cls, "conf": conf,
            "timestamp_sec": round(frame / 30, 2)}


def test_single_frame_false_positive_rejected():
    inc, _ = verify_incidents([det(0, 100, "D10_transverse_crack", 0.5)])
    assert inc == []


def test_other_class_in_same_frame_does_not_break_streak():
    d = [det(0, 10, "D00_longitudinal_crack", 0.6), det(1, 10, "D40_pothole", 0.8),
         det(2, 15, "D00_longitudinal_crack", 0.6), det(3, 15, "D40_pothole", 0.8)]
    inc, _ = verify_incidents(d, frame_interval=5)
    assert sorted(i["anomaly"] for i in inc) == ["D00_longitudinal_crack", "D40_pothole"]


def test_evidence_counts_distinct_frames():
    d = [det(0, 10, "D40_pothole", 0.6), det(1, 10, "D40_pothole", 0.7), det(2, 15, "D40_pothole", 0.8)]
    inc, _ = verify_incidents(d, frame_interval=5)
    assert inc[0]["evidence_frames"] == 2 and inc[0]["detections"] == 3


def test_two_boxes_in_one_frame_is_not_a_streak():
    d = [det(0, 10, "D40_pothole", 0.6), det(1, 10, "D40_pothole", 0.7)]
    inc, _ = verify_incidents(d, frame_interval=5)
    assert inc == []


def test_large_frame_interval_still_works():
    d = [det(0, 0, "D40_pothole", 0.6), det(1, 15, "D40_pothole", 0.7)]
    inc, _ = verify_incidents(d, frame_interval=15)
    assert len(inc) == 1


def test_scene_cut_splits_groups():
    d = [det(0, 10, "D40_pothole", 0.6, seg=1), det(1, 15, "D40_pothole", 0.7, seg=2)]
    inc, _ = verify_incidents(d, frame_interval=5)
    assert inc == []


def test_repair_is_set_aside():
    d = [det(0, 10, "repair", 0.6), det(1, 15, "repair", 0.7)]
    inc, ign = verify_incidents(d, frame_interval=5)
    assert inc == [] and len(ign) == 1 and ign[0]["status"] == "IGNORED"


def test_low_confidence_filtered():
    d = [det(0, 10, "D40_pothole", 0.3), det(1, 15, "D40_pothole", 0.35)]
    assert verify_incidents(d, min_conf=0.4)[0] == []
    assert len(verify_incidents(d, min_conf=0.25)[0]) == 1


def test_context_weather_rules():
    inc = {"incident_id": "X", "anomaly": "D00_longitudinal_crack"}
    assert context_agent(inc, {"condition": "Clear"})["risk_level"] == "Normal"
    assert context_agent(inc, {"condition": "Raining"})["risk_level"] == "Increased"
    assert context_agent(inc, {"condition": "Recently rained (wet road)"})["risk_level"] == "Increased"
    unk = context_agent(inc, {"condition": "Unknown"})
    assert unk["risk_level"] == "Normal" and "not provided" in unk["conclusion"]
    assert context_agent(inc, {"condition": "Rain"})["weather_condition"] == "Raining"   # OWM alias


def test_severity_formula_unchanged_from_v1():
    # v1 saved output: RA-0000, conf 0.519, 5 frames, Normal -> 33.6 MEDIUM
    s = severity_agent({"incident_id": "A", "avg_confidence": 0.519, "evidence_frames": 5}, {"risk_level": "Normal"})
    assert s["severity_score"] == 33.6 and s["severity"] == "MEDIUM"
    assert sum(s["breakdown"].values()) == s["severity_score"]


def test_final_record():
    inc = {"incident_id": "A", "anomaly": "D40_pothole", "evidence_frames": 3, "avg_confidence": 0.9,
           "location_label": "Segment 1 · 1.0–2.0 s"}
    ctx = context_agent(inc, {"condition": "Raining", "source": "manual"})
    sev = severity_agent(inc, ctx)
    f = build_final_incident(inc, ctx, sev, routing_agent(inc, sev))
    assert f["department"] == "Road Maintenance" and f["weather_source"] == "manual"
    assert f["severity"] == "HIGH"   # 27 + 6 + 25 + 8 = 66


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
