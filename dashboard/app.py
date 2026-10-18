"""
Road Anomaly Detection - Live Pipeline Dashboard (v2)

Run from the project root:
    streamlit run dashboard/app.py

v1 -> v2
* inputs redesigned around the real user: video, weather at recording time,
  detection preset, frame interval. API key / model path moved to config.
* no GPS: incidents are located by segment + video timestamp (timeline instead of map)
* scene-cut detection, duplicate-frame skipping, per-class verification
* per-factor severity breakdown, repaired sections listed separately
* every run saved to outputs/runs/<timestamp>/ + CSV / JSON download
"""
import os
import sys
import time

import cv2
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import settings
from agents.context_agent import WEATHER_OPTIONS
from pipeline.run_pipeline import run_pipeline, new_run_dir, STEPS, CSV_FIELDS

st.set_page_config(page_title="Road Anomaly — Live Pipeline v2", layout="wide")

SEVERITY_COLORS = {"CRITICAL": "#D32F2F", "HIGH": "#F57C00", "MEDIUM": "#FBC02D", "LOW": "#388E3C"}
STEP_COLORS = ["#818CF8", "#38BDF8", "#F43F5E", "#06B6D4", "#C084FC", "#FB923C", "#4ADE80"]


def render_html(html_str):
    """Strip line breaks/indentation so Streamlit markdown doesn't treat HTML as code."""
    return "".join(line.strip() for line in html_str.splitlines())


st.markdown("""
<style>
.stApp { background-color: #F4F6F9; }
[data-testid="stSidebar"] { background-color: #151B2B !important; border-right: 1px solid #1F2740 !important; }
[data-testid="stSidebar"] * { color: #F8FAFC !important; }
[data-testid="stSidebar"] label { color: #CBD5E1 !important; font-size: 13px !important; font-weight: 600 !important; }
[data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] .stCaption { color: #94A3B8 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #0F172A !important; }
[data-testid="stSidebar"] input { color: #0F172A !important; }
.sidebar-section-title { color: #3D7BFF !important; font-size: 11px !important; font-weight: 800 !important;
  letter-spacing: 1.2px !important; text-transform: uppercase !important; margin-top: 18px !important;
  margin-bottom: 8px !important; padding-bottom: 4px !important; border-bottom: 1px solid #1F2740 !important; }
div.stButton > button[kind="primary"] { background-color: #D9482B !important; border-color: #D9482B !important;
  color: #FFFFFF !important; font-weight: 700 !important; border-radius: 8px !important; }
div.stButton > button[kind="primary"]:hover { background-color: #B83A20 !important; border-color: #B83A20 !important; }
div.stButton > button[kind="secondary"] { background-color: #1F2740 !important; border-color: #334155 !important;
  color: #CBD5E1 !important; border-radius: 8px !important; }
.summary-card { background: #FFFFFF; border: 1px solid #E1E5EC; border-radius: 12px; padding: 20px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.03); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- session state
for key, default in (("result", None), ("console_logs", [])):
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------- html helpers
def top_bar_html(status_text, status_type="idle"):
    styles = {
        "running": ("rgba(245,158,11,0.15)", "#F59E0B", "#F59E0B", "#FDE68A"),
        "complete": ("rgba(34,197,94,0.15)", "#22C55E", "#22C55E", "#86EFAC"),
        "idle": ("rgba(148,163,184,0.15)", "#64748B", "#22C55E", "#E2E8F0"),
    }
    bg, border, dot, txt = styles.get(status_type, styles["idle"])
    return render_html(f'''
    <div style="background-color:#151B2B;padding:14px 24px;border-radius:12px;margin-bottom:20px;display:flex;
      align-items:center;justify-content:space-between;border:1px solid #1F2740;font-family:system-ui,sans-serif;">
      <div style="display:flex;align-items:center;gap:12px;">
        <span style="color:#FFFFFF;font-size:20px;font-weight:700;">Road Anomaly Detection</span>
        <span style="color:#94A3B8;font-size:13px;font-weight:500;padding-left:12px;border-left:1px solid #334155;">
          Live Pipeline Dashboard · v2</span>
      </div>
      <div style="background-color:{bg};border:1px solid {border};padding:6px 14px;border-radius:20px;display:flex;
        align-items:center;gap:8px;">
        <span style="width:8px;height:8px;border-radius:50%;background-color:{dot};display:inline-block;"></span>
        <span style="color:{txt};font-size:12px;font-weight:600;font-family:monospace;">{status_text}</span>
      </div>
    </div>''')


def stepper_html(active_step=0):
    """active_step: 0 = idle, 1..N = running that step, N+1 = all done."""
    n = len(STEPS)
    parts = ['<div style="display:flex;align-items:center;justify-content:space-between;background:#FFFFFF;'
             'padding:18px 24px;border-radius:12px;border:1px solid #E1E5EC;margin-bottom:20px;'
             'font-family:system-ui,sans-serif;">']
    for idx, label in enumerate(STEPS, start=1):
        if active_step > n or (0 < active_step and idx < active_step):
            bg, fg, content, lc, lw, sh = "#22C55E", "#FFF", "&#10003;", "#15803D", "600", "none"
        elif idx == active_step:
            bg, fg, content, lc, lw, sh = "#3D7BFF", "#FFF", str(idx), "#1D4ED8", "700", "0 0 0 4px rgba(61,123,255,0.25)"
        else:
            bg, fg, content, lc, lw, sh = "#E2E8F0", "#64748B", str(idx), "#94A3B8", "500", "none"
        parts.append(f'<div style="display:flex;flex-direction:column;align-items:center;min-width:64px;">'
                     f'<div style="width:32px;height:32px;border-radius:50%;background-color:{bg};color:{fg};display:flex;'
                     f'align-items:center;justify-content:center;font-size:13px;font-weight:700;box-shadow:{sh};">{content}</div>'
                     f'<span style="margin-top:6px;font-size:11px;color:{lc};font-weight:{lw};text-align:center;">{label}</span></div>')
        if idx < n:
            done = active_step > n or (active_step > 0 and active_step > idx)
            parts.append(f'<div style="flex:1;height:3px;background-color:{"#22C55E" if done else "#E2E8F0"};'
                         f'margin:0 4px;transform:translateY(-10px);"></div>')
    parts.append("</div>")
    return "".join(parts)


def console_html(logs):
    head = ('<div style="display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #1E293B;'
            'padding-bottom:10px;margin-bottom:14px;"><span style="color:#38BDF8;font-weight:700;font-size:13px;">'
            'AGENT CONSOLE LOG</span><span style="background:#1E293B;color:#94A3B8;font-size:10px;padding:2px 8px;'
            'border-radius:4px;font-weight:600;">{}</span></div>')
    box = ('<div style="background-color:#0F172A;border:1px solid #1E293B;border-radius:12px;padding:16px;'
           'font-family:monospace;color:#E2E8F0;min-height:380px;max-height:700px;overflow-y:auto;">')
    if not logs:
        return box + head.format("IDLE") + ('<div style="color:#64748B;margin-top:20px;font-style:italic;">'
                                            '[System idle] Upload a video and click "Process video".</div></div>')
    items = []
    for log in logs:
        c = log.get("color", "#38BDF8")
        detail = (f'<div style="color:#94A3B8;font-size:11px;margin-top:4px;padding-left:8px;border-left:2px solid #334155;'
                  f'white-space:pre-wrap;">{log["detail"]}</div>') if log.get("detail") else ""
        items.append(f'<div style="margin-bottom:12px;padding-bottom:10px;border-bottom:1px dashed #1E293B;"><div>'
                     f'<span style="background:{c}20;color:{c};border:1px solid {c}40;font-size:10px;font-weight:700;'
                     f'padding:2px 6px;border-radius:4px;margin-right:8px;">[{log["tag"]}]</span>'
                     f'<span style="color:#F8FAFC;font-size:12px;font-weight:600;">{log["title"]}</span></div>{detail}</div>')
    return box + head.format("LIVE FEED") + "".join(items) + "</div>"


@st.cache_resource(show_spinner=False)
def load_model(path):
    from ultralytics import YOLO
    return YOLO(path)


# ---------------------------------------------------------------- layout
top_bar = st.empty()
stepper = st.empty()
if st.session_state.result is not None:
    top_bar.markdown(top_bar_html("Pipeline complete", "complete"), unsafe_allow_html=True)
    stepper.markdown(stepper_html(len(STEPS) + 1), unsafe_allow_html=True)
else:
    top_bar.markdown(top_bar_html("Pipeline idle — ready"), unsafe_allow_html=True)
    stepper.markdown(stepper_html(0), unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-section-title">1. Upload</div>', unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Dashcam video", type=["mp4", "mov", "avi", "mkv"])

    st.markdown('<div class="sidebar-section-title">2. Recording conditions</div>', unsafe_allow_html=True)
    weather_condition = st.selectbox(
        "Weather when the video was recorded", WEATHER_OPTIONS, index=0,
        help="Used by the Context Agent. 'Recently rained' = rain has stopped but the road is still wet. "
             "Leave 'Unknown' if you don't know - no weather adjustment is applied then.")
    recorded_on = st.date_input("Recording date (optional, for the report)", value=None)

    st.markdown('<div class="sidebar-section-title">3. Detection</div>', unsafe_allow_html=True)
    preset = st.radio("Detection mode", list(settings.CONFIDENCE_PRESETS),
                      index=list(settings.CONFIDENCE_PRESETS).index(settings.DEFAULT_PRESET),
                      help="High recall finds more defects but more false alarms; "
                           "high precision reports fewer, more certain defects.")
    conf_threshold = settings.CONFIDENCE_PRESETS[preset]
    frame_interval = st.slider("Frame interval (process every Nth frame)", 1, 15, settings.DEFAULT_FRAME_INTERVAL,
                               help="Lower = more frames analysed (slower, more evidence per defect).")
    detect_cuts = st.checkbox("Detect scene cuts (joined clips)", value=True,
                              help="Splits the video where one clip ends and another begins, so defects on "
                                   "different roads are never merged.")
    st.caption(f"Model: {os.path.relpath(settings.MODEL_PATH, PROJECT_ROOT)} (set in config/settings.py)")

    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("Process video", type="primary", use_container_width=True)
    if st.session_state.result is not None:
        if st.button("Clear results", type="secondary", use_container_width=True):
            st.session_state.result = None
            st.session_state.console_logs = []
            st.rerun()

if run_button and not uploaded_video:
    st.error("Upload a video first.")

col_left, col_right = st.columns([62, 38])
with col_right:
    st.markdown("<h4 style='color:#1E293B;margin-bottom:12px;font-weight:700;'>Agent Console</h4>", unsafe_allow_html=True)
    console = st.empty()
    console.markdown(console_html(st.session_state.console_logs), unsafe_allow_html=True)


def log(tag, title, detail="", color="#38BDF8"):
    st.session_state.console_logs.append({"tag": tag, "title": title, "detail": detail, "color": color})
    console.markdown(console_html(st.session_state.console_logs), unsafe_allow_html=True)


# ---------------------------------------------------------------- run
if uploaded_video and run_button:
    if not os.path.exists(settings.MODEL_PATH):
        st.error(f"Model not found at {settings.MODEL_PATH}. Update MODEL_PATH in config/settings.py.")
        st.stop()
    st.session_state.console_logs = []
    run_dir = new_run_dir()
    ext = os.path.splitext(uploaded_video.name)[1] or ".mp4"
    video_path = os.path.join(run_dir, "input" + ext)
    with open(video_path, "wb") as fh:
        fh.write(uploaded_video.getbuffer())

    with col_left:
        progress_slot = st.empty()
        status_box = st.status("Pipeline execution details", expanded=True)

    def on_step(i, name, info):
        stepper.markdown(stepper_html(i + 2), unsafe_allow_html=True)
        nxt = STEPS[i + 1] if i + 1 < len(STEPS) else "done"
        top_bar.markdown(top_bar_html(f"Running — step {min(i + 2, len(STEPS))} of {len(STEPS)} ({nxt})", "running"),
                         unsafe_allow_html=True)
        c = STEP_COLORS[i]
        with status_box:
            st.write(f"**Step {i + 1} — {name}** ✓")
        if i == 0:
            segs = info["segments"]
            log("SCENE CUTS", f"{len(segs)} segment(s) found",
                "\n".join(f"Segment {s['segment_id']}: {s['start_sec']:.1f}–{s['end_sec']:.1f} s" for s in segs), c)
        elif i == 1:
            log("FRAMES", f"{info['frames_saved']} frames extracted",
                f"Interval: every {info['frame_interval']} frames | duplicate frames skipped: {info['duplicates_skipped']}", c)
            with status_box:
                st.write("**Step 3 — Detection** running…")
        elif i == 2:
            progress_slot.empty()
            log("YOLO DETECTION", f"{info['raw_detections']} raw detections",
                f"Mode: {preset} | model: {os.path.basename(settings.MODEL_PATH)}", c)
        elif i == 3:
            v, ig = info["verified"], info["ignored"]
            detail = "\n".join(f"• {x['incident_id']}: {x['anomaly']} — {x['evidence_frames']} frames, "
                               f"avg conf {x['avg_confidence']} ({x['location_label']})" for x in v) \
                or "No detection streak met the minimum evidence."
            if ig:
                detail += f"\n{len(ig)} repaired section(s) set aside"
            log("VERIFICATION AGENT", f"{len(v)} verified incident(s)", detail, c)
        elif i == 4:
            log("CONTEXT AGENT", f"Weather: {weather_condition} (entered manually)" if weather_condition != "Unknown"
                else "Weather: Unknown (no adjustment)",
                "\n".join(f"• {x['incident_id']}: {x['conclusion']} (risk {x['risk_level']})" for x in info["context"]), c)
        elif i == 5:
            log("SEVERITY AGENT", "Severity scores calculated",
                "\n".join(f"• {x['incident_id']}: {x['severity_score']} → {x['severity']}  "
                          f"(conf {x['breakdown']['confidence']} + evid {x['breakdown']['evidence']} + "
                          f"risk {x['breakdown']['context_risk']} + recur {x['breakdown']['recurrence']})"
                          for x in info["severity"]), c)
        elif i == 6:
            log("ROUTING AGENT", "Department routing done",
                "\n".join(f"• {x['incident_id']}: {x['department']} → {x['recommended_action']}"
                          for x in info["final"]), c)

    bar = progress_slot.progress(0.0, text="Detecting…")
    top_bar.markdown(top_bar_html(f"Running — step 1 of {len(STEPS)} (Scene cuts)", "running"), unsafe_allow_html=True)
    stepper.markdown(stepper_html(1), unsafe_allow_html=True)
    try:
        model = load_model(settings.MODEL_PATH)
        result = run_pipeline(
            video_path, model, weather_condition=weather_condition, conf_threshold=conf_threshold,
            frame_interval=frame_interval, detect_cuts=detect_cuts, run_dir=run_dir,
            recorded_at=str(recorded_on) if recorded_on else None,
            on_step=on_step, on_progress=lambda d, t: bar.progress(d / t, text=f"Detecting… frame {d}/{t}"))
    except Exception as e:
        status_box.update(label="Pipeline failed", state="error")
        st.exception(e)
        st.stop()
    result["meta"]["preset"] = preset
    st.session_state.result = result
    status_box.update(label="Pipeline execution details — complete", state="complete", expanded=False)
    stepper.markdown(stepper_html(len(STEPS) + 1), unsafe_allow_html=True)
    top_bar.markdown(top_bar_html("Pipeline complete", "complete"), unsafe_allow_html=True)

# ---------------------------------------------------------------- results
res = st.session_state.result
if res is not None:
    finals = res["final_incidents"]
    raw = res["raw_detections"]
    ignored = res["ignored_repaired"]

    with col_left:
        st.markdown("<h4 style='color:#1E293B;font-weight:700;'>Detection funnel</h4>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Frames analysed", res["frame_stats"]["frames_saved"],
                  help=f"{res['frame_stats']['duplicates_skipped']} duplicate frames skipped")
        m2.metric("Raw detections", len(raw))
        m3.metric("Verified incidents", len(finals))
        m4.metric("Repaired sections", len(ignored), help="`repair` class - no action needed")

        if len(res["segments"]) > 1:
            st.markdown("<h4 style='color:#1E293B;font-weight:700;'>Segments</h4>", unsafe_allow_html=True)
            seg_rows = []
            for s in res["segments"]:
                seg_inc = [f for f in finals if f["segment_id"] == s["segment_id"]]
                seg_rows.append({"Segment": s["segment_id"], "From (s)": s["start_sec"], "To (s)": s["end_sec"],
                                 "Raw detections": sum(1 for d in raw if d.get("segment_id") == s["segment_id"]),
                                 "Incidents": len(seg_inc),
                                 "Worst severity": next((sv for sv in SEVERITY_COLORS
                                                         if any(f["severity"] == sv for f in seg_inc)), "—")})
            st.dataframe(pd.DataFrame(seg_rows), hide_index=True, use_container_width=True)

        if not finals:
            st.info("No verified incidents in this video at the current detection mode / frame interval.")
        else:
            st.markdown("<h4 style='color:#1E293B;font-weight:700;'>Incident timeline</h4>", unsafe_allow_html=True)
            import altair as alt
            tl = pd.DataFrame([{
                "incident": f["incident_id"], "anomaly": f["anomaly"], "severity": f["severity"],
                "score": f["severity_score"], "start": f["time_range_sec"][0],
                "end": max(f["time_range_sec"][1], f["time_range_sec"][0] + 0.2), "location": f["location"],
            } for f in finals])
            chart = alt.Chart(tl).mark_bar(size=16, cornerRadius=3).encode(
                x=alt.X("start:Q", title="Video time (s)",
                        scale=alt.Scale(domain=[0, res["segments"][-1]["end_sec"]])),
                x2="end:Q",
                y=alt.Y("anomaly:N", title=None),
                color=alt.Color("severity:N", scale=alt.Scale(domain=list(SEVERITY_COLORS),
                                                              range=list(SEVERITY_COLORS.values()))),
                tooltip=["incident", "anomaly", "severity", "score", "location"],
            ).properties(height=60 + 40 * tl["anomaly"].nunique())
            cut_times = [s["start_sec"] for s in res["segments"][1:]]
            if cut_times:
                rules = alt.Chart(pd.DataFrame({"t": cut_times})).mark_rule(strokeDash=[4, 4], color="#94A3B8").encode(x="t:Q")
                chart = chart + rules
            st.altair_chart(chart, use_container_width=True)

            st.markdown("<h4 style='color:#1E293B;margin:16px 0;font-weight:700;'>Detected defect gallery</h4>",
                        unsafe_allow_html=True)
            det_by_id = {d["det_id"]: d for d in raw}
            grid = st.columns(2)
            for idx, f in enumerate(finals):
                color = SEVERITY_COLORS.get(f["severity"], "#757575")
                b = f.get("score_breakdown", {})
                with grid[idx % 2]:
                    st.markdown(render_html(f'''
                    <div style="background:#FFFFFF;border:1px solid #E1E5EC;border-top:4px solid {color};border-radius:10px;
                      padding:14px 16px 10px 16px;margin-bottom:12px;">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                        <div><span style="font-weight:800;font-size:14px;color:#0F172A;">{f['incident_id']}</span>
                          <span style="color:#475569;font-size:12px;font-weight:600;margin-left:6px;">{f['anomaly']}</span></div>
                        <span style="background-color:{color}18;color:{color};border:1px solid {color}40;padding:2px 8px;
                          border-radius:12px;font-size:11px;font-weight:700;">{f['severity']} ({f['severity_score']})</span>
                      </div>
                      <div style="font-size:11px;color:#64748B;margin-bottom:4px;">📍 {f['location']} · {f['evidence_frames']} frames ·
                        avg conf {f['avg_confidence']}</div>
                      <div style="font-size:12px;color:#475569;margin-bottom:4px;">{f['context_conclusion']}</div>
                      <div style="font-size:11px;color:#64748B;margin-bottom:4px;font-family:monospace;">
                        score = conf {b.get('confidence')} + evidence {b.get('evidence')} + risk {b.get('context_risk')}
                        + recurrence {b.get('recurrence')}*</div>
                      <div style="font-size:11px;color:#1E293B;font-weight:600;"><b>{f['department']}:</b> {f['recommended_action']}</div>
                    </div>'''), unsafe_allow_html=True)

                    dets = [det_by_id[i] for i in f.get("member_det_ids", []) if i in det_by_id]
                    seen, shown = set(), []
                    for d in sorted(dets, key=lambda d: -d["conf"]):
                        if d["frame_path"] not in seen:
                            seen.add(d["frame_path"])
                            shown.append(d)
                        if len(shown) == 3:
                            break
                    if shown:
                        cols = st.columns(len(shown))
                        for c, d in zip(cols, sorted(shown, key=lambda d: d["frame"])):
                            img = cv2.imread(d["frame_path"])
                            if img is None:
                                continue
                            x1, y1, x2, y2 = map(int, d["bbox"])
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(img, f"{d['class']} {d['conf']:.2f}", (x1, max(y1 - 10, 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            with c:
                                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                         caption=f"{d['timestamp_sec']:.1f} s", use_container_width=True)
            st.caption("* Recurrence is fixed at 1 detection until incidents are stored across runs (future work).")

        if ignored:
            with st.expander(f"Repaired sections set aside ({len(ignored)})"):
                st.dataframe(pd.DataFrame([{"id": i["incident_id"], "location": i["location_label"],
                                            "frames": i["evidence_frames"], "avg conf": i["avg_confidence"]}
                                           for i in ignored]), hide_index=True, use_container_width=True)

    with col_right:
        counts = {sv: sum(1 for f in finals if f["severity"] == sv) for sv in SEVERITY_COLORS}
        rows_html = "".join(
            f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;font-size:12px;">'
            f'<div style="display:flex;align-items:center;gap:8px;"><span style="width:10px;height:10px;border-radius:2px;'
            f'background-color:{SEVERITY_COLORS[sv]};display:inline-block;"></span><span style="font-weight:600;color:#334155;">'
            f'{sv}</span></div><span style="font-weight:700;color:#0F172A;background:#F1F5F9;padding:2px 8px;border-radius:10px;">'
            f'{n}</span></div>' for sv, n in counts.items() if n)
        meta = res["meta"]
        w = meta["weather"]
        st.markdown(render_html(f'''
        <div class="summary-card" style="margin-top:20px;">
          <div style="font-size:13px;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.5px;">Summary</div>
          <div style="font-size:38px;font-weight:800;color:#3D7BFF;margin:6px 0 2px 0;">{len(finals)}</div>
          <div style="font-size:12px;color:#475569;font-weight:600;margin-bottom:14px;">Verified incidents</div>
          <hr style="border:none;border-top:1px solid #F1F5F9;margin:10px 0;">
          {rows_html or '<div style="font-size:12px;color:#64748B;">No incidents</div>'}
          <hr style="border:none;border-top:1px solid #F1F5F9;margin:10px 0;">
          <div style="font-size:11px;color:#64748B;line-height:1.7;">
            Weather: <b>{w['condition']}</b> ({w['source']})<br>
            Mode: <b>{meta.get('preset', meta['conf_threshold'])}</b> · every {meta['frame_interval']} frames<br>
            Recorded: <b>{meta.get('recorded_at') or 'not given'}</b><br>
            Location: by video timestamp (no GPS in this version)
          </div>
        </div>'''), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        csv_bytes = pd.DataFrame(finals, columns=CSV_FIELDS).to_csv(index=False).encode("utf-8") if finals else b""
        st.download_button("⬇ Download incidents (CSV)", csv_bytes, file_name="incidents.csv",
                           mime="text/csv", use_container_width=True, disabled=not finals)
        run_json = os.path.join(res["run_dir"], "run.json")
        if os.path.exists(run_json):
            with open(run_json, "rb") as fh:
                st.download_button("⬇ Download full run (JSON)", fh.read(), file_name="run.json",
                                   mime="application/json", use_container_width=True)
        st.caption(f"Run saved to: {os.path.relpath(res['run_dir'], PROJECT_ROOT)}")
        with st.expander("Inspect raw agent JSON output"):
            st.json(finals)

elif not uploaded_video:
    with col_left:
        st.info("Upload a video, choose the weather at recording time, then click **Process video**.")
