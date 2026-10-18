"""
Severity Agent

The formula and weights are UNCHANGED from v1 so results stay comparable:

    score = avg_confidence * W_CONF                      (max 30)  detection certainty
          + min(evidence_frames, EVIDENCE_CAP) * W_EVID  (max 20)  strength of evidence
          + RISK_WEIGHTS[risk_level]                     (0 / 25)  context risk
          + min(repetition_count, REPEAT_CAP) * W_REPEAT (max 40)  recurrence

    >= 70 CRITICAL | >= 50 HIGH | >= 30 MEDIUM | else LOW

v2 additions (no behaviour change)
* weights live in one WEIGHTS dict so evaluation/severity_sensitivity.py can vary them
* returns a per-factor `breakdown`, so every score is explainable on the dashboard
* NOTE: repetition_count is fixed at 1 until incidents are stored across runs
  (future work), so the recurrence term is currently a constant +8.
* `evidence_frames` now means distinct frames (see verification_agent v2).
"""

WEIGHTS = {
    "W_CONF": 30,
    "W_EVID": 2,
    "EVIDENCE_CAP": 10,
    "RISK_INCREASED": 25,
    "W_REPEAT": 8,
    "REPEAT_CAP": 5,
}
THRESHOLDS = {"CRITICAL": 70, "HIGH": 50, "MEDIUM": 30}


def categorise(score, thresholds=THRESHOLDS):
    if score >= thresholds["CRITICAL"]:
        return "CRITICAL"
    if score >= thresholds["HIGH"]:
        return "HIGH"
    if score >= thresholds["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def severity_agent(incident, context_result, repetition_count=1, weights=None, thresholds=None):
    """Numerical severity score, category and per-factor breakdown for an incident."""
    w = dict(WEIGHTS, **(weights or {}))
    breakdown = {
        "confidence": incident["avg_confidence"] * w["W_CONF"],
        "evidence": min(incident["evidence_frames"], w["EVIDENCE_CAP"]) * w["W_EVID"],
        "context_risk": w["RISK_INCREASED"] if context_result["risk_level"] == "Increased" else 0,
        "recurrence": min(repetition_count, w["REPEAT_CAP"]) * w["W_REPEAT"],
    }
    score = sum(breakdown.values())
    return {
        "incident_id": incident["incident_id"],
        "severity_score": round(score, 1),
        "severity": categorise(score, thresholds or THRESHOLDS),
        "breakdown": {k: round(v, 1) for k, v in breakdown.items()},
    }


if __name__ == "__main__":
    print(severity_agent({"incident_id": "RA-00010-D40", "avg_confidence": 0.85, "evidence_frames": 4},
                         {"risk_level": "Increased"}))
