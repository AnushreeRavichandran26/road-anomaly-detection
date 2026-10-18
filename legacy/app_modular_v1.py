import os
import sys
import tempfile
import cv2
import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

# Ensure project root is in sys.path when launched from any working directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.verification_agent import verify_incidents
from agents.context_agent import context_agent
from agents.severity_agent import severity_agent
from agents.routing_agent import routing_agent, build_final_incident
from pipeline.frame_extractor import extract_frames
from pipeline.gps_tagger import simulate_gps_from_start
from pipeline.weather_lookup import get_weather
from pipeline.yolo_inference import run_detection
from ultralytics import YOLO

st.set_page_config(page_title="Road Anomaly — Live Pipeline (Modular)", layout="wide")

# =================================================================
# HELPER TO CLEAN HTML (PREVENTS MARKDOWN INDENTED CODE BLOCKS)
# =================================================================
def render_html(html_str):
    """Strip linebreaks and leading spaces so markdown parser doesn't render HTML as code blocks."""
    return "".join(line.strip() for line in html_str.splitlines())

# =================================================================
# CUSTOM CSS FOR MOCKUP DESIGN & RIGHT CONSOLE PANEL
# =================================================================
st.markdown("""
<style>
/* Main app light background */
.stApp {
    background-color: #F4F6F9;
}

/* Sidebar Dark Navy Styling */
[data-testid="stSidebar"] {
    background-color: #151B2B !important;
    border-right: 1px solid #1F2740 !important;
}
[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] .stSelectbox label, 
[data-testid="stSidebar"] .stNumberInput label, 
[data-testid="stSidebar"] .stSlider label, 
[data-testid="stSidebar"] .stTextInput label, 
[data-testid="stSidebar"] .stFileUploader label {
    color: #CBD5E1 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #94A3B8 !important;
}

/* Sidebar Section Headers */
.sidebar-section-title {
    color: #3D7BFF !important;
    font-size: 11px !important;
    font-weight: 800 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    margin-top: 18px !important;
    margin-bottom: 8px !important;
    padding-bottom: 4px !important;
    border-bottom: 1px solid #1F2740 !important;
}

/* Sidebar Primary Action Button (Process Video) */
div.stButton > button[kind="primary"] {
    background-color: #D9482B !important;
    border-color: #D9482B !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    box-shadow: 0 4px 12px rgba(217, 72, 43, 0.25) !important;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #B83A20 !important;
    border-color: #B83A20 !important;
}

/* Sidebar Secondary Button (Clear Results) */
div.stButton > button[data-testid="baseButton-secondary"],
div.stButton > button[kind="secondary"] {
    background-color: #1F2740 !important;
    border-color: #334155 !important;
    color: #CBD5E1 !important;
    border-radius: 8px !important;
}
div.stButton > button[data-testid="baseButton-secondary"]:hover,
div.stButton > button[kind="secondary"]:hover {
    background-color: #334155 !important;
    color: #FFFFFF !important;
}

/* Summary Card styling */
.summary-card {
    background: #FFFFFF;
    border: 1px solid #E1E5EC;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
}
</style>
""", unsafe_allow_html=True)

# =================================================================
# SESSION STATE — keeps results visible across Streamlit re-runs
# =================================================================
if "final_incidents" not in st.session_state:
    st.session_state.final_incidents = None
    st.session_state.raw_detections = None
    st.session_state.verified = None
    st.session_state.console_logs = []

SEVERITY_COLORS = {
    "CRITICAL": "#D32F2F",
    "HIGH": "#F57C00",
    "MEDIUM": "#FBC02D",
    "LOW": "#388E3C",
}

# =================================================================
# HTML TEMPLATE HELPERS (TOP BAR, STEPPER, AGENT CONSOLE)
# =================================================================

def get_top_bar_html(status_text="Pipeline idle — ready", status_type="idle"):
    if status_type == "running":
        badge_bg = "rgba(245, 158, 11, 0.15)"
        badge_border = "#F59E0B"
        dot_color = "#F59E0B"
        text_color = "#FDE68A"
    elif status_type == "complete":
        badge_bg = "rgba(34, 197, 94, 0.15)"
        badge_border = "#22C55E"
        dot_color = "#22C55E"
        text_color = "#86EFAC"
    else:
        badge_bg = "rgba(148, 163, 184, 0.15)"
        badge_border = "#64748B"
        dot_color = "#22C55E"
        text_color = "#E2E8F0"

    raw = f'''
    <div style="background-color: #151B2B; padding: 14px 24px; border-radius: 12px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; border: 1px solid #1F2740; font-family: system-ui, -apple-system, sans-serif;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="color: #FFFFFF; font-size: 20px; font-weight: 700; letter-spacing: -0.3px;">Road Anomaly Detection</span>
            <span style="color: #94A3B8; font-size: 13px; font-weight: 500; padding-left: 12px; border-left: 1px solid #334155;">Live Pipeline Dashboard</span>
        </div>
        <div style="background-color: {badge_bg}; border: 1px solid {badge_border}; padding: 6px 14px; border-radius: 20px; display: flex; align-items: center; gap: 8px;">
            <span style="width: 8px; height: 8px; border-radius: 50%; background-color: {dot_color}; display: inline-block;"></span>
            <span style="color: {text_color}; font-size: 12px; font-weight: 600; font-family: monospace;">{status_text}</span>
        </div>
    </div>
    '''
    return render_html(raw)


def get_stepper_html(active_step=0):
    steps = [
        ("1", "Frames"),
        ("2", "GPS"),
        ("3", "Weather"),
        ("4", "Detection"),
        ("5", "Verification"),
        ("6", "Context"),
        ("7", "Severity"),
        ("8", "Routing"),
    ]

    html_parts = ['<div style="display: flex; align-items: center; justify-content: space-between; background: #FFFFFF; padding: 18px 24px; border-radius: 12px; border: 1px solid #E1E5EC; box-shadow: 0 2px 8px rgba(0,0,0,0.03); margin-bottom: 20px; font-family: system-ui, -apple-system, sans-serif;">']

    for idx, (num, label) in enumerate(steps, start=1):
        if active_step > 8 or (active_step > 0 and idx < active_step):
            circle_bg = "#22C55E"
            circle_color = "#FFFFFF"
            circle_content = "&#10003;"  # Checkmark ✓
            circle_shadow = "none"
            label_color = "#15803D"
            label_weight = "600"
        elif idx == active_step:
            circle_bg = "#3D7BFF"
            circle_color = "#FFFFFF"
            circle_content = num
            circle_shadow = "0 0 0 4px rgba(61, 123, 255, 0.25)"
            label_color = "#1D4ED8"
            label_weight = "700"
        else:
            circle_bg = "#E2E8F0"
            circle_color = "#64748B"
            circle_content = num
            circle_shadow = "none"
            label_color = "#94A3B8"
            label_weight = "500"

        node_html = f'''
        <div style="display: flex; flex-direction: column; align-items: center; min-width: 60px;">
            <div style="width: 32px; height: 32px; border-radius: 50%; background-color: {circle_bg}; color: {circle_color}; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; box-shadow: {circle_shadow}; transition: all 0.3s ease;">
                {circle_content}
            </div>
            <span style="margin-top: 6px; font-size: 11px; color: {label_color}; font-weight: {label_weight}; text-align: center;">{label}</span>
        </div>
        '''
        html_parts.append(node_html)

        if idx < 8:
            line_color = "#22C55E" if (active_step > 8 or (active_step > 0 and active_step > idx)) else "#E2E8F0"
            line_html = f'''
            <div style="flex: 1; height: 3px; background-color: {line_color}; margin: 0 4px; transform: translateY(-10px); transition: background-color 0.3s ease;"></div>
            '''
            html_parts.append(line_html)

    html_parts.append('</div>')
    return render_html("".join(html_parts))


def render_console_html(logs):
    """Render a dark telemetry panel for agent intermediate outputs on the right side."""
    if not logs:
        raw = '''
        <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 12px; padding: 16px; font-family: monospace; min-height: 420px; color: #64748B; font-size: 12px;">
            <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 10px; margin-bottom: 14px;">
                <span style="color: #38BDF8; font-weight: 700; font-size: 13px;">AGENT CONSOLE LOG</span>
                <span style="background: #1E293B; color: #94A3B8; font-size: 10px; padding: 2px 8px; border-radius: 4px; font-weight: 600;">IDLE</span>
            </div>
            <div style="color: #64748B; margin-top: 20px; font-style: italic;">[System Idle] Upload video & click "Process video" to stream live intermediate agent decisions...</div>
        </div>
        '''
        return render_html(raw)

    log_items = []
    for log in logs:
        color = log.get("color", "#38BDF8")
        tag_html = f'<span style="background: {color}20; color: {color}; border: 1px solid {color}40; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; margin-right: 8px;">[{log.get("tag", "INFO")}]</span>'
        title_html = f'<span style="color: #F8FAFC; font-size: 12px; font-weight: 600;">{log.get("title", "")}</span>'
        detail_html = f'<div style="color: #94A3B8; font-size: 11px; margin-top: 4px; padding-left: 8px; border-left: 2px solid #334155; white-space: pre-wrap; font-family: monospace;">{log.get("detail", "")}</div>' if log.get("detail") else ''
        
        item = f'<div style="margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px dashed #1E293B;"><div>{tag_html}{title_html}</div>{detail_html}</div>'
        log_items.append(item)

    raw = f'''
    <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 12px; padding: 16px; font-family: monospace; color: #E2E8F0; max-height: 700px; overflow-y: auto;">
        <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 10px; margin-bottom: 14px;">
            <span style="color: #38BDF8; font-weight: 700; font-size: 13px;">AGENT CONSOLE LOG</span>
            <span style="background: #22C55E22; color: #4ADE80; border: 1px solid #22C55E44; font-size: 10px; padding: 2px 8px; border-radius: 4px; font-weight: 700;">LIVE FEED</span>
        </div>
        {"".join(log_items)}
    </div>
    '''
    return render_html(raw)


# =================================================================
# TOP BAR AND STEPPER PLACEHOLDERS (CREATED ONCE AT THE TOP)
# =================================================================

top_bar_placeholder = st.empty()
stepper_placeholder = st.empty()

# Render initial top bar & stepper states
if st.session_state.final_incidents is not None:
    top_bar_placeholder.markdown(get_top_bar_html("Pipeline complete", "complete"), unsafe_allow_html=True)
    stepper_placeholder.markdown(get_stepper_html(9), unsafe_allow_html=True)
else:
    top_bar_placeholder.markdown(get_top_bar_html("Pipeline idle — ready", "idle"), unsafe_allow_html=True)
    stepper_placeholder.markdown(get_stepper_html(0), unsafe_allow_html=True)

# =================================================================
# SIDEBAR — GROUPED INPUT SECTIONS
# =================================================================

with st.sidebar:
    st.markdown('<div class="sidebar-section-title">1. UPLOAD</div>', unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Dashcam video", type=["mp4", "mov", "avi", "mkv"])

    st.markdown('<div class="sidebar-section-title">2. STARTING GPS</div>', unsafe_allow_html=True)
    start_lat = st.number_input("Latitude", value=13.0827, format="%.6f")
    start_lon = st.number_input("Longitude", value=80.2707, format="%.6f")
    assumed_speed_kmh = st.slider("Assumed vehicle speed (km/h)", 5, 80, 30)
    st.caption("Position estimated assuming constant speed. Replace with real GPS log in production.")

    st.markdown('<div class="sidebar-section-title">3. WEATHER</div>', unsafe_allow_html=True)
    weather_api_key = st.text_input("OpenWeatherMap API key", type="password")

    st.markdown('<div class="sidebar-section-title">4. DETECTION</div>', unsafe_allow_html=True)
    model_path = st.text_input("Model path", value="models/best.pt")
    frame_interval = st.slider("Frame interval (process every Nth frame)", 1, 30, 5)
    conf_threshold = st.slider("Detection confidence threshold", 0.1, 0.9, 0.4)

    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("Process video", type="primary", use_container_width=True)

    if st.session_state.final_incidents is not None:
        if st.button("Clear results", type="secondary", use_container_width=True):
            st.session_state.final_incidents = None
            st.session_state.raw_detections = None
            st.session_state.verified = None
            st.session_state.console_logs = []
            st.rerun()

if not uploaded_video and run_button:
    st.error("Upload a video first.")

# =================================================================
# MAIN LAYOUT SPLIT: LEFT CONTENT (MAP & GALLERY) / RIGHT AGENT CONSOLE
# =================================================================

col_left, col_right = st.columns([62, 38])

with col_right:
    st.markdown("<h4 style='color: #1E293B; margin-bottom: 12px; font-weight: 700;'>Agent Console</h4>", unsafe_allow_html=True)
    console_placeholder = st.empty()
    console_placeholder.markdown(render_console_html(st.session_state.console_logs), unsafe_allow_html=True)

# =================================================================
# PIPELINE PROCESSING (SINGLE SCRIPT EXECUTION WITH LIVE UPDATES)
# =================================================================

if uploaded_video and run_button:
    # Reset console logs
    st.session_state.console_logs = []

    with col_left:
        progress_placeholder = st.empty()
        status_box = st.status("Pipeline Execution Details", expanded=True)

        with status_box:
            # -------------------------------------------------------------
            # STEP 1 — FRAME EXTRACTION
            # -------------------------------------------------------------
            top_bar_placeholder.markdown(get_top_bar_html("Running — Step 1 of 8 (Frames)", "running"), unsafe_allow_html=True)
            stepper_placeholder.markdown(get_stepper_html(1), unsafe_allow_html=True)

            st.write("**Step 1 — Extracting frames from video...**")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                tmp_video.write(uploaded_video.read())
                tmp_video_path = tmp_video.name

            frames_dir = tempfile.mkdtemp(prefix="frames_")
            frame_records, fps = extract_frames(tmp_video_path, frames_dir, frame_interval=frame_interval)
            st.write(f"Extracted {len(frame_records)} frames (source FPS: {fps:.1f})")

            st.session_state.console_logs.append({
                "tag": "FRAMES",
                "color": "#38BDF8",
                "title": f"Extracted {len(frame_records)} frames",
                "detail": f"Source FPS: {fps:.1f} | Frame Interval: {frame_interval}"
            })
            console_placeholder.markdown(render_console_html(st.session_state.console_logs), unsafe_allow_html=True)

            # -------------------------------------------------------------
            # STEP 2 — GPS ESTIMATION
            # -------------------------------------------------------------
            top_bar_placeholder.markdown(get_top_bar_html("Running — Step 2 of 8 (GPS)", "running"), unsafe_allow_html=True)
            stepper_placeholder.markdown(get_stepper_html(2), unsafe_allow_html=True)

            st.write("**Step 2 — Estimating GPS per frame...**")
            frame_records = simulate_gps_from_start(
                frame_records, start_lat, start_lon, assumed_speed_kmh, fps, frame_interval
            )
            st.write(f"GPS range: {frame_records[0]['gps']} → {frame_records[-1]['gps']}")

            st.session_state.console_logs.append({
                "tag": "GPS TAGGER",
                "color": "#818CF8",
                "title": f"Estimated route coordinates from ({start_lat:.4f}, {start_lon:.4f})",
                "detail": f"Start GPS: {frame_records[0]['gps']}\nEnd GPS:   {frame_records[-1]['gps']}"
            })
            console_placeholder.markdown(render_console_html(st.session_state.console_logs), unsafe_allow_html=True)

            # -------------------------------------------------------------
            # STEP 3 — WEATHER LOOKUP
            # -------------------------------------------------------------
            top_bar_placeholder.markdown(get_top_bar_html("Running — Step 3 of 8 (Weather)", "running"), unsafe_allow_html=True)
            stepper_placeholder.markdown(get_stepper_html(3), unsafe_allow_html=True)

            st.write("**Step 3 — Fetching weather for location...**")
            weather = get_weather(start_lat, start_lon, weather_api_key)
            st.write(f"Weather data: {weather}")
            for rec in frame_records:
                rec["weather"] = weather

            st.session_state.console_logs.append({
                "tag": "WEATHER API",
                "color": "#FBBF24",
                "title": f"Weather: {weather.get('condition', 'Unknown')} ({weather.get('description', '')})",
                "detail": f"Temp: {weather.get('temp_c')}°C | Humidity: {weather.get('humidity')}%"
            })
            console_placeholder.markdown(render_console_html(st.session_state.console_logs), unsafe_allow_html=True)

            # -------------------------------------------------------------
            # STEP 4 — YOLO DETECTION
            # -------------------------------------------------------------
            top_bar_placeholder.markdown(get_top_bar_html("Running — Step 4 of 8 (Detection)", "running"), unsafe_allow_html=True)
            stepper_placeholder.markdown(get_stepper_html(4), unsafe_allow_html=True)

            st.write("**Step 4 — Running YOLO detection model...**")
            if not os.path.exists(model_path):
                st.error(f"Model not found at {model_path}. Check sidebar path.")
                st.stop()

            model = YOLO(model_path)
            st.write(f"Model classes: {model.names}")

            progress = progress_placeholder.progress(0)
            def update_progress(curr, total):
                progress.progress(curr / total)

            raw_detections = run_detection(model, frame_records, conf_threshold=conf_threshold, progress_callback=update_progress)
            st.write(f"Total raw detections found: {len(raw_detections)}")

            st.session_state.console_logs.append({
                "tag": "YOLO DETECTION",
                "color": "#F43F5E",
                "title": f"Detection loop completed: {len(raw_detections)} raw detections",
                "detail": f"Model: {model_path} | Conf Threshold: {conf_threshold}"
            })
            console_placeholder.markdown(render_console_html(st.session_state.console_logs), unsafe_allow_html=True)

            # -------------------------------------------------------------
            # STEP 5 — VERIFICATION AGENT
            # -------------------------------------------------------------
            top_bar_placeholder.markdown(get_top_bar_html("Running — Step 5 of 8 (Verification)", "running"), unsafe_allow_html=True)
            stepper_placeholder.markdown(get_stepper_html(5), unsafe_allow_html=True)

            st.write("**Step 5 — Running Verification Agent...**")
            verified = verify_incidents(raw_detections, min_conf=conf_threshold)
            st.write(f"Verified incidents: {len(verified)} (from {len(raw_detections)} raw detections)")
            for inc in verified:
                st.write(f"  - {inc['incident_id']}: {inc['anomaly']} ({inc['evidence_frames']} frames, avg conf {inc['avg_confidence']})")

            st.session_state.console_logs.append({
                "tag": "VERIFICATION AGENT",
                "color": "#06B6D4",
                "title": f"Merged raw detections into {len(verified)} verified incidents",
                "detail": "\n".join([f"• {inc['incident_id']}: {inc['anomaly']} ({inc['evidence_frames']} frames, avg conf {inc['avg_confidence']})" for inc in verified]) if verified else "No detections met streak threshold."
            })
            console_placeholder.markdown(render_console_html(st.session_state.console_logs), unsafe_allow_html=True)

            # -------------------------------------------------------------
            # STEP 6 — CONTEXT AGENT
            # -------------------------------------------------------------
            top_bar_placeholder.markdown(get_top_bar_html("Running — Step 6 of 8 (Context)", "running"), unsafe_allow_html=True)
            stepper_placeholder.markdown(get_stepper_html(6), unsafe_allow_html=True)
            st.write("**Step 6 — Running Context Agent...**")

            # -------------------------------------------------------------
            # STEP 7 — SEVERITY AGENT
            # -------------------------------------------------------------
            top_bar_placeholder.markdown(get_top_bar_html("Running — Step 7 of 8 (Severity)", "running"), unsafe_allow_html=True)
            stepper_placeholder.markdown(get_stepper_html(7), unsafe_allow_html=True)
            st.write("**Step 7 — Running Severity Agent...**")

            # -------------------------------------------------------------
            # STEP 8 — ROUTING AGENT & FINAL REPORT BUILD
            # -------------------------------------------------------------
            top_bar_placeholder.markdown(get_top_bar_html("Running — Step 8 of 8 (Routing)", "running"), unsafe_allow_html=True)
            stepper_placeholder.markdown(get_stepper_html(8), unsafe_allow_html=True)
            st.write("**Step 8 — Running Routing Agent & constructing final incidents...**")

            final_incidents = []
            context_log_details = []
            severity_log_details = []
            routing_log_details = []

            for inc in verified:
                ctx = context_agent(inc, weather, repetition_count=1)
                sev = severity_agent(inc, ctx, repetition_count=1)
                rte = routing_agent(inc, sev)
                final = build_final_incident(inc, ctx, sev, rte)
                final_incidents.append(final)
                st.write(f"  - {final['incident_id']}: {final['severity']} ({final['severity_score']}) → {final['department']} — {final['recommended_action']}")

                context_log_details.append(f"• {final['incident_id']}: {final['context_conclusion']} (Risk: {final['risk_level']})")
                severity_log_details.append(f"• {final['incident_id']}: Score {final['severity_score']} → {final['severity']}")
                routing_log_details.append(f"• {final['incident_id']}: Dept -> {final['department']} | Directive -> {final['recommended_action']}")

            st.session_state.console_logs.append({
                "tag": "CONTEXT AGENT",
                "color": "#C084FC",
                "title": "Contextual Hazard Analysis Completed",
                "detail": "\n".join(context_log_details) if context_log_details else "No verified incidents to evaluate."
            })
            st.session_state.console_logs.append({
                "tag": "SEVERITY AGENT",
                "color": "#FB923C",
                "title": "Severity Scores Calculated",
                "detail": "\n".join(severity_log_details) if severity_log_details else "No verified incidents to score."
            })
            st.session_state.console_logs.append({
                "tag": "ROUTING AGENT",
                "color": "#4ADE80",
                "title": "Department Routing Directives Created",
                "detail": "\n".join(routing_log_details) if routing_log_details else "No verified incidents to route."
            })
            console_placeholder.markdown(render_console_html(st.session_state.console_logs), unsafe_allow_html=True)

            # Store in session state for persistence across re-runs
            st.session_state.final_incidents = final_incidents
            st.session_state.raw_detections = raw_detections
            st.session_state.verified = verified

            # Final complete state for Top Bar and Stepper
            stepper_placeholder.markdown(get_stepper_html(9), unsafe_allow_html=True)
            top_bar_placeholder.markdown(get_top_bar_html("Pipeline complete", "complete"), unsafe_allow_html=True)

        # Collapse status box when complete
        status_box.update(label="Pipeline Execution Details — Complete", state="complete", expanded=False)

# =================================================================
# RESULTS DISPLAY (LEFT COLUMN: MAP + GALLERY / RIGHT COLUMN: SUMMARY)
# =================================================================

if st.session_state.final_incidents is not None:
    final_incidents = st.session_state.final_incidents
    raw_detections = st.session_state.raw_detections
    verified = st.session_state.verified

    with col_left:
        if not final_incidents:
            st.info("No verified incidents found in this video at the current confidence/interval settings.")
        else:
            # MAP
            st.markdown("<h4 style='color: #1E293B; margin-bottom: 12px; font-weight: 700;'>Incident Map</h4>", unsafe_allow_html=True)
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

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #1E293B; margin-bottom: 16px; font-weight: 700;'>Detected Defect Gallery</h4>", unsafe_allow_html=True)

            incident_lookup = {inc["incident_id"]: inc for inc in verified}

            # 2-COLUMN GRID OF INCIDENT CARDS inside left column
            grid_cols = st.columns(2)

            for idx, final in enumerate(final_incidents):
                inc = incident_lookup[final["incident_id"]]
                color = SEVERITY_COLORS.get(final["severity"], "#757575")

                with grid_cols[idx % 2]:
                    card_header_html = render_html(f'''
                    <div style="background: #FFFFFF; border: 1px solid #E1E5EC; border-top: 4px solid {color}; border-radius: 10px; padding: 14px 16px 10px 16px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <div>
                                <span style="font-weight: 800; font-size: 15px; color: #0F172A;">{final['incident_id']}</span>
                                <span style="color: #475569; font-size: 12px; font-weight: 600; margin-left: 6px;">{final['anomaly']}</span>
                            </div>
                            <span style="background-color: {color}18; color: {color}; border: 1px solid {color}40; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 700;">
                                {final['severity']} ({final['severity_score']})
                            </span>
                        </div>
                        <div style="font-size: 12px; color: #475569; margin-bottom: 4px;">{final['context_conclusion']}</div>
                        <div style="font-size: 11px; color: #1E293B; font-weight: 600;"><b>Action:</b> {final['recommended_action']}</div>
                    </div>
                    ''')
                    st.markdown(card_header_html, unsafe_allow_html=True)

                    # Thumbnail sub-row inside card container
                    member_frame_numbers = set(final.get("member_frames", []))
                    relevant_dets = [d for d in raw_detections if d["frame"] in member_frame_numbers and d["class"] == final["anomaly"]]

                    shown_paths = set()
                    thumb_cols = st.columns(min(3, max(1, len(relevant_dets))))
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

                        with thumb_cols[col_idx % len(thumb_cols)]:
                            st.image(img_rgb, caption=f"Frame {det['frame']}", use_container_width=True)
                        col_idx += 1

                    st.markdown("<br>", unsafe_allow_html=True)

    with col_right:
        # PIPELINE SUMMARY METRICS CARD BELOW THE AGENT CONSOLE
        departments = list(set(i["department"] for i in final_incidents if i.get("department")))
        dept_str = ", ".join(departments) if departments else "None"

        summary_card_html = render_html(f'''
        <div class="summary-card" style="margin-top: 20px;">
            <div style="font-size: 13px; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px;">Summary</div>
            <div style="font-size: 38px; font-weight: 800; color: #3D7BFF; margin: 6px 0 2px 0;">{len(final_incidents)}</div>
            <div style="font-size: 12px; color: #475569; font-weight: 600; margin-bottom: 14px;">Verified Incidents Detected</div>
            <hr style="border: none; border-top: 1px solid #F1F5F9; margin: 10px 0;">
            <div style="font-size: 12px; font-weight: 700; color: #1E293B; margin-bottom: 8px;">Severity Breakdown</div>
        ''' + "".join([
            f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; font-size: 12px;"><div style="display: flex; align-items: center; gap: 8px;"><span style="width: 10px; height: 10px; border-radius: 2px; background-color: {SEVERITY_COLORS.get(sev, "#757575")}; display: inline-block;"></span><span style="font-weight: 600; color: #334155;">{sev}</span></div><span style="font-weight: 700; color: #0F172A; background: #F1F5F9; padding: 2px 8px; border-radius: 10px;">{sum(1 for i in final_incidents if i["severity"] == sev)}</span></div>'
            for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW") if sum(1 for i in final_incidents if i["severity"] == sev)
        ]) + f'''
            <hr style="border: none; border-top: 1px solid #F1F5F9; margin: 10px 0;">
            <div style="font-size: 11px; color: #64748B; font-weight: 600;">Departments Involved</div>
            <div style="font-size: 12px; color: #0F172A; font-weight: 700; margin-top: 4px;">{dept_str}</div>
        </div>
        ''')
        st.markdown(summary_card_html, unsafe_allow_html=True)

        with st.expander("Inspect Raw Agent JSON Output", expanded=False):
            st.json(final_incidents)

elif not uploaded_video:
    with col_left:
        st.info("Upload a video and set the starting GPS point in the sidebar, then click **Process video**.")
