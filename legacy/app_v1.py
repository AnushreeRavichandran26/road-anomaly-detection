import streamlit as st
import cv2
import os
import tempfile
import math
import time
from math import radians, sin, cos, sqrt, atan2

import requests
import folium
from streamlit_folium import st_folium
from PIL import Image

st.set_page_config(page_title="Road Anomaly — Live Pipeline", layout="wide")

# =================================================================
# SESSION STATE — keeps results visible across Streamlit re-runs
# (e.g. the map widget triggers a background re-run; without this,
# results would disappear immediately after appearing)
# =================================================================
if "final_incidents" not in st.session_state:
    st.session_state.final_incidents = None
    st.session_state.raw_detections = None
    st.session_state.verified = None

# =================================================================
# AGENT FUNCTIONS (self-contained — same logic as the notebooks)
# =================================================================

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
    }


def verify_incidents(detections, min_conf=0.4, max_frame_gap=10, max_gps_dist=25, min_streak=2):
    detections = sorted(detections, key=lambda d: d["frame"])
    incidents = []
    current_group = []

    for det in detections:
        if det["conf"] < min_conf:
            continue
        if not current_group:
            current_group = [det]
            continue

        last = current_group[-1]
        same_class = det["class"] == last["class"]
        close_frame = (det["frame"] - last["frame"]) <= max_frame_gap
        close_gps = gps_distance_meters(det["gps"], last["gps"]) <= max_gps_dist

        if same_class and close_frame and close_gps:
            current_group.append(det)
        else:
            if len(current_group) >= min_streak:
                incidents.append(build_incident(current_group))
            current_group = [det]

    if len(current_group) >= min_streak:
        incidents.append(build_incident(current_group))

    return incidents


def context_agent(incident, weather, repetition_count=1):
    anomaly = incident["anomaly"]
    condition = weather.get("condition", "Unknown")
    is_rainy = condition in ("Rain", "Drizzle", "Thunderstorm")

    context = {
        "incident_id": incident["incident_id"],
        "anomaly": anomaly,
        "weather_condition": condition,
        "repetition_count": repetition_count,
        "conclusion": None,
        "risk_level": "Normal",
    }

    if "water" in anomaly.lower():
        if is_rainy:
            context["conclusion"] = "Water accumulation likely related to current rainfall"
        elif repetition_count > 1:
            context["conclusion"] = "Possible persistent drainage issue"
            context["risk_level"] = "Increased"
        else:
            context["conclusion"] = "Water presence unexplained by weather — needs inspection"
            context["risk_level"] = "Increased"
    elif "crack" in anomaly.lower():
        if is_rainy:
            context["conclusion"] = "Crack may worsen faster due to water seepage"
            context["risk_level"] = "Increased"
        elif repetition_count > 1:
            context["conclusion"] = "Recurring crack — monitor for expansion"
            context["risk_level"] = "Increased"
        else:
            context["conclusion"] = "Standard crack, no weather-related urgency"
    elif "pothole" in anomaly.lower():
        if is_rainy:
            context["conclusion"] = "Pothole with active rain — increased vehicle hazard"
            context["risk_level"] = "Increased"
        elif repetition_count > 1:
            context["conclusion"] = "Recurring pothole — likely worsening, needs prioritized repair"
            context["risk_level"] = "Increased"
        else:
            context["conclusion"] = "Pothole detected, standard road hazard"
    else:
        context["conclusion"] = "General road anomaly detected"

    return context


def severity_agent(incident, context_result, repetition_count=1):
    score = 0
    score += incident["avg_confidence"] * 30
    score += min(incident["evidence_frames"], 10) * 2
    risk_weights = {"Normal": 0, "Increased": 25}
    score += risk_weights.get(context_result["risk_level"], 0)
    score += min(repetition_count, 5) * 8

    if score >= 70:
        severity = "CRITICAL"
    elif score >= 50:
        severity = "HIGH"
    elif score >= 30:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return {"incident_id": incident["incident_id"], "severity_score": round(score, 1), "severity": severity}


DEPARTMENT_MAP = {
    "D00_longitudinal_crack": "Road Maintenance",
    "D10_transverse_crack": "Road Maintenance",
    "D20_alligator_crack": "Road Maintenance",
    "D40_pothole": "Road Maintenance",
    "repair": None,
}


def routing_agent(incident, severity_result):
    anomaly = incident["anomaly"]
    department = DEPARTMENT_MAP.get(anomaly, "General Public Works")

    if department is None:
        return {"incident_id": incident["incident_id"], "status": "IGNORED", "reason": "Already-repaired section"}

    if severity_result["severity"] == "CRITICAL":
        action = "Immediate inspection and repair required within 24 hours"
    elif severity_result["severity"] == "HIGH":
        action = "Priority inspection within 3-5 days"
    elif severity_result["severity"] == "MEDIUM":
        action = "Schedule routine inspection within 2 weeks"
    else:
        action = "Log for periodic maintenance review"

    return {
        "incident_id": incident["incident_id"],
        "department": department,
        "severity": severity_result["severity"],
        "recommended_action": action,
    }


def build_final_incident(incident, context_result, severity_result, routing_result):
    return {
        "incident_id": incident["incident_id"],
        "anomaly": incident["anomaly"],
        "status": routing_result.get("status", "VERIFIED"),
        "gps": incident["gps"],
        "weather": context_result["weather_condition"],
        "evidence_frames": incident["evidence_frames"],
        "avg_confidence": incident["avg_confidence"],
        "context_conclusion": context_result["conclusion"],
        "risk_level": context_result["risk_level"],
        "severity": severity_result["severity"],
        "severity_score": severity_result["severity_score"],
        "department": routing_result.get("department", "N/A"),
        "recommended_action": routing_result.get("recommended_action", routing_result.get("reason", "N/A")),
        "member_frames": incident.get("member_frames", []),
    }


# =================================================================
# WEATHER
# =================================================================

def get_weather(lat, lon, api_key):
    if not api_key:
        return {"condition": "Unknown", "description": "No API key provided", "temp_c": None, "humidity": None}
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        return {
            "condition": data["weather"][0]["main"],
            "description": data["weather"][0]["description"],
            "temp_c": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
        }
    except Exception as e:
        return {"condition": "Unknown", "description": f"fetch failed: {e}", "temp_c": None, "humidity": None}


# =================================================================
# GPS SIMULATION FROM A SINGLE STARTING POINT
# =================================================================

def simulate_gps_from_start(frame_records, start_lat, start_lon, assumed_speed_kmh, fps, frame_interval):
    """
    Only a starting GPS point is available (no real GPS log / metadata track),
    so position is estimated frame-by-frame assuming constant speed and a
    fixed heading (north). This is a deliberate, documented approximation —
    a real deployment would replace this with an actual GPS log matched by
    timestamp, without changing anything downstream.
    """
    speed_m_s = assumed_speed_kmh * 1000 / 3600
    seconds_per_extracted_frame = frame_interval / fps if fps > 0 else 0
    meters_per_frame_step = speed_m_s * seconds_per_extracted_frame
    meters_per_degree_lat = 111320.0

    for idx, rec in enumerate(frame_records):
        delta_lat = (meters_per_frame_step * idx) / meters_per_degree_lat
        rec["gps"] = (round(start_lat + delta_lat, 6), round(start_lon, 6))
    return frame_records


# =================================================================
# FRAME EXTRACTION
# =================================================================

def extract_frames(video_path, output_dir, frame_interval=5):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = 0
    saved_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{frame_count:05d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_frames.append({
                "frame_number": frame_count,
                "timestamp_sec": round(frame_count / fps, 2),
                "frame_path": frame_path,
            })
        frame_count += 1

    cap.release()
    return saved_frames, fps


# =================================================================
# STREAMLIT UI
# =================================================================

st.title("Road Anomaly Detection — Live Pipeline")
st.caption("Upload a dashcam video → frames → weather → detection → verification → context → severity → routing")

SEVERITY_COLORS = {
    "CRITICAL": "#d32f2f",
    "HIGH": "#f57c00",
    "MEDIUM": "#fbc02d",
    "LOW": "#388e3c",
}

with st.sidebar:
    st.header("Inputs")
    uploaded_video = st.file_uploader("Dashcam video", type=["mp4", "mov", "avi", "mkv"])

    st.subheader("Starting GPS point")
    start_lat = st.number_input("Latitude", value=13.0827, format="%.6f")
    start_lon = st.number_input("Longitude", value=80.2707, format="%.6f")
    assumed_speed_kmh = st.slider("Assumed vehicle speed (km/h)", 5, 80, 30)
    st.caption("Only a starting point is available, so position along the route is estimated from assumed speed. Replace with a real GPS log for production use.")

    st.subheader("Weather")
    weather_api_key = st.text_input("OpenWeatherMap API key", type="password")

    st.subheader("Detection")
    model_path = st.text_input("Model path", value="models/best.pt")
    frame_interval = st.slider("Frame interval (process every Nth frame)", 1, 30, 5)
    conf_threshold = st.slider("Detection confidence threshold", 0.1, 0.9, 0.4)

    run_button = st.button("Process video", type="primary", use_container_width=True)

    if st.session_state.final_incidents is not None:
        if st.button("Clear results", use_container_width=True):
            st.session_state.final_incidents = None
            st.session_state.raw_detections = None
            st.session_state.verified = None
            st.rerun()

if not uploaded_video and run_button:
    st.error("Upload a video first.")

# =================================================================
# PROCESSING — runs once per button click, saves everything to
# session_state so it survives the map widget's background re-run
# =================================================================
if uploaded_video and run_button:

    status_box = st.status("Running pipeline...", expanded=True)

    with status_box:
        st.write("**Step 1 — Extracting frames from video**")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_video.read())
            tmp_video_path = tmp_video.name

        frames_dir = tempfile.mkdtemp(prefix="frames_")
        frame_records, fps = extract_frames(tmp_video_path, frames_dir, frame_interval=frame_interval)
        st.write(f"Extracted {len(frame_records)} frames (source fps: {fps:.1f})")

        st.write("**Step 2 — Estimating GPS per frame from starting point**")
        frame_records = simulate_gps_from_start(
            frame_records, start_lat, start_lon, assumed_speed_kmh, fps, frame_interval
        )
        st.write(f"GPS range: {frame_records[0]['gps']} → {frame_records[-1]['gps']}")

        st.write("**Step 3 — Fetching weather for starting location**")
        weather = get_weather(start_lat, start_lon, weather_api_key)
        st.write(f"Weather: {weather}")
        for rec in frame_records:
            rec["weather"] = weather

        st.write("**Step 4 — Running detection model**")
        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}. Check the path in the sidebar.")
            st.stop()

        from ultralytics import YOLO
        model = YOLO(model_path)
        st.write(f"Model classes: {model.names}")

        raw_detections = []
        progress = st.progress(0)
        for i, rec in enumerate(frame_records):
            results = model.predict(rec["frame_path"], conf=conf_threshold, verbose=False)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                raw_detections.append({
                    "frame": rec["frame_number"],
                    "class": model.names[cls_id],
                    "conf": round(float(box.conf[0]), 3),
                    "bbox": box.xyxy[0].tolist(),
                    "gps": rec["gps"],
                    "weather": rec["weather"],
                    "frame_path": rec["frame_path"],
                })
            progress.progress((i + 1) / len(frame_records))
        st.write(f"Total raw detections: {len(raw_detections)}")

        st.write("**Step 5 — Verification Agent (merging repeated detections)**")
        verified = verify_incidents(raw_detections, min_conf=conf_threshold)
        st.write(f"Verified incidents: {len(verified)} (from {len(raw_detections)} raw detections)")
        for inc in verified:
            st.write(f"  - {inc['incident_id']}: {inc['anomaly']} ({inc['evidence_frames']} frames, avg conf {inc['avg_confidence']})")

        st.write("**Step 6-8 — Context, Severity, and Routing Agents**")
        final_incidents = []
        for inc in verified:
            ctx = context_agent(inc, weather, repetition_count=1)
            sev = severity_agent(inc, ctx, repetition_count=1)
            rte = routing_agent(inc, sev)
            final = build_final_incident(inc, ctx, sev, rte)
            final_incidents.append(final)
            st.write(f"  - {final['incident_id']}: {final['severity']} ({final['severity_score']}) → {final['department']} — {final['recommended_action']}")

        st.write("**Pipeline complete.**")

        # Save everything so it survives Streamlit re-runs (e.g. triggered by the map widget)
        st.session_state.final_incidents = final_incidents
        st.session_state.raw_detections = raw_detections
        st.session_state.verified = verified

    status_box.update(label="Pipeline complete", state="complete")

# =================================================================
# RESULTS — displayed from session_state, independent of the button
# click, so the map widget's re-run does not make results vanish
# =================================================================
if st.session_state.final_incidents is not None:
    final_incidents = st.session_state.final_incidents
    raw_detections = st.session_state.raw_detections
    verified = st.session_state.verified

    st.divider()
    st.subheader("Results")

    if not final_incidents:
        st.info("No verified incidents found in this video at the current confidence/interval settings.")
    else:
        col_map, col_summary = st.columns([2, 1])

        with col_map:
            st.markdown("#### Incident map")
            avg_lat = sum(i["gps"][0] for i in final_incidents) / len(final_incidents)
            avg_lon = sum(i["gps"][1] for i in final_incidents) / len(final_incidents)
            m = folium.Map(location=[avg_lat, avg_lon], zoom_start=16, tiles="CartoDB positron")
            for inc in final_incidents:
                lat, lon = inc["gps"]
                color = SEVERITY_COLORS.get(inc["severity"], "#757575")
                popup_html = (
                    f"<b>{inc['incident_id']}</b><br>{inc['anomaly']}<br>"
                    f"Severity: {inc['severity']} ({inc['severity_score']})<br>"
                    f"Dept: {inc['department']}<br>{inc['recommended_action']}"
                )
                folium.CircleMarker(
                    location=[lat, lon], radius=10, color=color, fill=True,
                    fill_color=color, fill_opacity=0.85,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{inc['incident_id']} — {inc['severity']}",
                ).add_to(m)
            st_folium(m, width=None, height=450, key="results_map")

        with col_summary:
            st.markdown("#### Summary")
            st.metric("Verified incidents", len(final_incidents))
            for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
                count = sum(1 for i in final_incidents if i["severity"] == sev)
                if count:
                    st.write(f"**{sev}:** {count}")

        st.divider()
        st.markdown("#### Frames with detected defects")

        incident_lookup = {inc["incident_id"]: inc for inc in verified}

        for final in final_incidents:
            inc = incident_lookup[final["incident_id"]]
            color = SEVERITY_COLORS.get(final["severity"], "#757575")

            with st.container(border=True):
                st.markdown(
                    f"**{final['incident_id']}** — {final['anomaly']} — "
                    f"<span style='color:{color}; font-weight:bold;'>{final['severity']}</span> "
                    f"({final['severity_score']}) — {final['department']}",
                    unsafe_allow_html=True,
                )
                st.caption(f"{final['context_conclusion']} | Recommended: {final['recommended_action']}")

                member_frame_numbers = set(final.get("member_frames", []))
                relevant_dets = [d for d in raw_detections if d["frame"] in member_frame_numbers and d["class"] == final["anomaly"]]

                shown_paths = set()
                cols = st.columns(min(4, max(1, len(relevant_dets))))
                col_idx = 0
                for det in relevant_dets:
                    if det["frame_path"] in shown_paths:
                        continue
                    shown_paths.add(det["frame_path"])

                    img = cv2.imread(det["frame_path"])
                    if img is None:
                        continue
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(img, f"{det['class']} {det['conf']:.2f}", (x1, max(y1 - 10, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    with cols[col_idx % len(cols)]:
                        st.image(img_rgb, caption=f"frame {det['frame']} (conf {det['conf']})", use_container_width=True)
                    col_idx += 1

elif not uploaded_video:
    st.info("Upload a video and set the starting GPS point in the sidebar, then click **Process video**.")