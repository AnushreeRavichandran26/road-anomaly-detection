"""
Routing Agent (v2)

Department mapping and severity -> action rules are UNCHANGED from v1.
v2: `repair` sections are filtered out earlier (verification_agent), and the
final record uses segment/timestamp location instead of GPS.
"""

DEPARTMENT_MAP = {
    "D00_longitudinal_crack": "Road Maintenance",
    "D10_transverse_crack": "Road Maintenance",
    "D20_alligator_crack": "Road Maintenance",
    "D40_pothole": "Road Maintenance",
    "repair": None,
}

ACTIONS = {
    "CRITICAL": "Immediate inspection and repair required within 24 hours",
    "HIGH": "Priority inspection within 3-5 days",
    "MEDIUM": "Schedule routine inspection within 2 weeks",
    "LOW": "Log for periodic maintenance review",
}


def routing_agent(incident, severity_result):
    """Determine responsible department and recommended action based on severity."""
    department = DEPARTMENT_MAP.get(incident["anomaly"], "General Public Works")
    if department is None:
        return {"incident_id": incident["incident_id"], "status": "IGNORED", "reason": "Already-repaired section"}
    return {
        "incident_id": incident["incident_id"],
        "department": department,
        "severity": severity_result["severity"],
        "recommended_action": ACTIONS[severity_result["severity"]],
    }


def build_final_incident(incident, context_result, severity_result, routing_result):
    """Combine output from all agents into a unified final incident report."""
    return {
        "incident_id": incident["incident_id"],
        "anomaly": incident["anomaly"],
        "status": routing_result.get("status", "VERIFIED"),
        "segment_id": incident.get("segment_id", 1),
        "location": incident.get("location_label", ""),
        "time_range_sec": incident.get("time_range_sec"),
        "frame_range": incident.get("frame_range"),
        "evidence_frames": incident["evidence_frames"],
        "detections": incident.get("detections", incident["evidence_frames"]),
        "avg_confidence": incident["avg_confidence"],
        "weather": context_result["weather_condition"],
        "weather_source": context_result.get("weather_source", "unknown"),
        "context_conclusion": context_result["conclusion"],
        "llm_explanation": context_result.get("llm_explanation"),
        "risk_level": context_result["risk_level"],
        "severity": severity_result["severity"],
        "severity_score": severity_result["severity_score"],
        "score_breakdown": severity_result.get("breakdown", {}),
        "department": routing_result.get("department", "N/A"),
        "recommended_action": routing_result.get("recommended_action", routing_result.get("reason", "N/A")),
        "member_frames": incident.get("member_frames", []),
        "member_det_ids": incident.get("member_det_ids", []),
    }


if __name__ == "__main__":
    inc = {"incident_id": "RA-00010-D40", "anomaly": "D40_pothole", "evidence_frames": 4,
           "avg_confidence": 0.85, "location_label": "Segment 1 · 0.3–0.5 s"}
    ctx = {"weather_condition": "Raining", "weather_source": "manual", "conclusion": "Hazard", "risk_level": "Increased"}
    sev = {"severity": "CRITICAL", "severity_score": 75.5}
    print(build_final_incident(inc, ctx, sev, routing_agent(inc, sev)))
