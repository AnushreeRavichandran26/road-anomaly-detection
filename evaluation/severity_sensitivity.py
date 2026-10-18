"""
Severity weight sensitivity analysis - answers "why 30 / 2 / 25 / 8?".

    python -m evaluation.severity_sensitivity outputs/runs/<run_folder> [--weather Raining]

For each weight, scale it by -20 % and +20 % (others fixed) and count how many
incidents change severity category. If few change, the ranking is robust to
the exact choice of weights; if many change, that weight needs justification.

With no run folder it uses a built-in synthetic set of incidents so the
analysis can still be demonstrated.
"""
import argparse
import json
import os

from agents.context_agent import context_agent
from agents.severity_agent import WEIGHTS, severity_agent

FACTORS = ["W_CONF", "W_EVID", "RISK_INCREASED", "W_REPEAT"]


def synthetic_incidents():
    out = []
    i = 0
    for conf in (0.45, 0.55, 0.65, 0.75, 0.85):
        for ev in (2, 4, 6, 10):
            for cls in ("D00_longitudinal_crack", "D40_pothole"):
                out.append({"incident_id": f"SYN-{i}", "anomaly": cls,
                            "avg_confidence": conf, "evidence_frames": ev})
                i += 1
    return out


def analyse(incidents, weather):
    ctxs = [context_agent(inc, weather) for inc in incidents]
    base = [severity_agent(inc, c)["severity"] for inc, c in zip(incidents, ctxs)]
    rows = []
    for f in FACTORS:
        for scale in (0.8, 1.2):
            w = {f: WEIGHTS[f] * scale}
            new = [severity_agent(inc, c, weights=w)["severity"] for inc, c in zip(incidents, ctxs)]
            changed = sum(1 for a, b in zip(base, new) if a != b)
            rows.append((f, scale, WEIGHTS[f] * scale, changed, len(incidents)))
    return base, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?")
    p.add_argument("--weather", default=None, help="override weather (default: the run's weather)")
    a = p.parse_args()

    if a.run_dir:
        with open(os.path.join(a.run_dir, "run.json"), encoding="utf-8") as fh:
            run = json.load(fh)
        incidents = run["verified_incidents"]
        weather = run["meta"]["weather"]
        src = a.run_dir
    else:
        incidents = synthetic_incidents()
        weather = {"condition": "Clear"}
        src = "synthetic incidents"
    if a.weather:
        weather = {"condition": a.weather, "source": "manual"}
    if not incidents:
        raise SystemExit("No verified incidents in this run.")

    base, rows = analyse(incidents, weather)
    print(f"Source: {src} | weather: {weather['condition']} | incidents: {len(incidents)}")
    print("Baseline categories:", {c: base.count(c) for c in ("CRITICAL", "HIGH", "MEDIUM", "LOW")})
    print(f"\n{'factor':<16}{'change':>8}{'new value':>11}{'incidents that change category':>34}")
    for f, s, v, ch, n in rows:
        print(f"{f:<16}{'-20%' if s < 1 else '+20%':>8}{v:>11.1f}{ch:>20} / {n}  ({100 * ch / n:.0f}%)")


if __name__ == "__main__":
    main()
