"""
Context Agent (v2)

Interprets a verified incident in light of the road-surface weather at the time
of recording. In v2 the weather is ENTERED BY THE USER (no GPS -> no online
lookup), and every result records where the weather came from.

Weather conditions
  Clear | Cloudy        -> dry road
  Raining               -> active rain
  Recently rained       -> rain has stopped but the road is still wet
  Unknown               -> no weather adjustment, and the report says so

v1 -> v2 changes
* new "Recently rained" condition: water seepage into cracks continues after rain stops
* "Unknown" weather is now stated explicitly instead of silently treated as dry
* `weather_source` recorded on every result (manual / api / unknown)
* removed the "water" branch: the model has no water class, so it could never run
"""

WEATHER_OPTIONS = ["Unknown", "Clear", "Cloudy", "Raining", "Recently rained (wet road)"]

# accept OpenWeatherMap-style labels too, in case the API module is used again
_ALIASES = {
    "rain": "Raining", "drizzle": "Raining", "thunderstorm": "Raining", "raining": "Raining",
    "clouds": "Cloudy", "cloudy": "Cloudy", "clear": "Clear",
    "recently rained (wet road)": "Recently rained (wet road)", "wet": "Recently rained (wet road)",
    "wet road": "Recently rained (wet road)", "recently rained": "Recently rained (wet road)",
}


def normalise_condition(condition):
    if not condition:
        return "Unknown"
    return _ALIASES.get(str(condition).strip().lower(), condition)


def context_agent(incident, weather, repetition_count=1):
    """Analyse environmental context (weather, repetition) for a verified road incident."""
    anomaly = incident["anomaly"]
    condition = normalise_condition(weather.get("condition"))
    source = weather.get("source", "manual" if condition != "Unknown" else "unknown")
    raining = condition == "Raining"
    wet = condition == "Recently rained (wet road)"
    unknown = condition == "Unknown"

    context = {
        "incident_id": incident["incident_id"],
        "anomaly": anomaly,
        "weather_condition": condition,
        "weather_source": source,
        "repetition_count": repetition_count,
        "conclusion": None,
        "risk_level": "Normal",
    }
    a = anomaly.lower()

    if "crack" in a:
        if raining:
            context["conclusion"] = "Crack under active rain — water seepage can widen it quickly"
            context["risk_level"] = "Increased"
        elif wet:
            context["conclusion"] = "Crack on a wet road — trapped water keeps seeping after rain stops"
            context["risk_level"] = "Increased"
        elif repetition_count > 1:
            context["conclusion"] = "Recurring crack — monitor for expansion"
            context["risk_level"] = "Increased"
        else:
            context["conclusion"] = "Standard crack, no weather-related urgency"
    elif "pothole" in a:
        if raining:
            context["conclusion"] = "Pothole with active rain — water hides its depth, higher vehicle hazard"
            context["risk_level"] = "Increased"
        elif wet:
            context["conclusion"] = "Pothole on a wet road — may be water-filled and harder to see"
            context["risk_level"] = "Increased"
        elif repetition_count > 1:
            context["conclusion"] = "Recurring pothole — likely worsening, needs prioritised repair"
            context["risk_level"] = "Increased"
        else:
            context["conclusion"] = "Pothole detected, standard road hazard"
    else:
        context["conclusion"] = "General road anomaly detected"

    if unknown:
        context["conclusion"] += " (weather not provided — no weather adjustment applied)"

    return context


def context_agent_llm(incident, weather, repetition_count=1, model_name="llama3.2:3b"):
    """
    Hybrid layer: rule-based context_agent() decides; a local Ollama model only
    rephrases the decision for the dashboard. Unchanged from v1 apart from the
    weather-source line in the prompt.
    """
    r = context_agent(incident, weather, repetition_count)
    prompt = f"""You are a road maintenance assistant. Given this structured incident data, write a single clear, professional sentence explaining the situation for a government dashboard. Do not add information not given below.

Incident ID: {r['incident_id']}
Anomaly type: {r['anomaly']}
Weather condition: {r['weather_condition']} (source: {r['weather_source']})
Times detected: {r['repetition_count']}
Preliminary conclusion: {r['conclusion']}
Risk level: {r['risk_level']}

Write only the explanation sentence, nothing else."""
    try:
        import ollama
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        r["llm_explanation"] = response["message"]["content"].strip()
    except Exception as e:
        r["llm_explanation"] = f"[LLM offline/unavailable: {e}] {r['conclusion']}"
    return r


if __name__ == "__main__":
    print(context_agent({"incident_id": "RA-00010-D40", "anomaly": "D40_pothole"},
                        {"condition": "Recently rained (wet road)", "source": "manual"}))
