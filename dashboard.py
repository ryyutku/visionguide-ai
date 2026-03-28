"""
VisualAid Detection System — Streamlit Dashboard
=================================================
Runs with dummy data out of the box.
To connect your real data, replace the functions inside:
    get_detections()  YOLOv8 model output
    get_ultrasonic_distance() HC-SR04 GPIO reading
    get_system_stats()  psutil readings from the Pi



Install:
    pip install streamlit pandas plotly

Run:
    streamlit run dashboard.py
"""

import time
import random
import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisualAid Detection Dashboard",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label {
        color: #8b8fa8 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #e8eaf0 !important;
        font-size: 28px !important;
        font-weight: 600 !important;
    }

    /* Section headers */
    .section-header {
        color: #8b8fa8;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid #2a2d3a;
    }

    /* Detection cards */
    .det-card {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .det-name { color: #e8eaf0; font-size: 15px; font-weight: 500; }
    .det-dir  { color: #8b8fa8; font-size: 12px; }
    .det-dist { color: #e8eaf0; font-size: 14px; font-weight: 600; }

    /* Priority badges */
    .badge {
        display: inline-block;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-high   { background: #3d1515; color: #f87171; }
    .badge-medium { background: #2d2010; color: #fbbf24; }
    .badge-low    { background: #0f2d1a; color: #4ade80; }

    /* Alert log */
    .alert-item {
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 6px;
        font-size: 13px;
        display: flex;
        gap: 10px;
        align-items: flex-start;
    }
    .alert-danger { background: #1f0f0f; border-left: 3px solid #ef4444; }
    .alert-warn   { background: #1f1a0f; border-left: 3px solid #f59e0b; }
    .alert-ok     { background: #0f1f14; border-left: 3px solid #22c55e; }
    .alert-time   { color: #8b8fa8; font-size: 11px; min-width: 42px; }
    .alert-msg    { color: #c8cad8; }

    /* System status row */
    .sys-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 0;
        border-bottom: 1px solid #1e2130;
        font-size: 13px;
    }
    .sys-label { color: #8b8fa8; }
    .status-on  { color: #4ade80; font-weight: 500; }
    .status-off { color: #f87171; font-weight: 500; }
    .status-warn{ color: #fbbf24; font-weight: 500; }

    /* Direction indicator */
    .dir-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 8px;
        margin: 8px 0;
    }
    .dir-cell {
        text-align: center;
        padding: 14px 6px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        border: 1px solid #2a2d3a;
        color: #8b8fa8;
        background: #1a1d27;
    }
    .dir-cell.active-warn {
        background: #2d2010;
        border-color: #f59e0b;
        color: #fbbf24;
    }
    .dir-cell.active-danger {
        background: #1f0f0f;
        border-color: #ef4444;
        color: #f87171;
    }

    /* Hide streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    /* Dataframe tweaks */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  DUMMY DATA LAYER
#  ↓↓↓ REPLACE THESE FUNCTIONS WITH YOUR REAL DATA SOURCES ↓↓↓
# ─────────────────────────────────────────────────────────────

OBJECT_POOL = [
    {"name": "Chair",    "priority": "high"},
    {"name": "Person",   "priority": "high"},
    {"name": "Table",    "priority": "high"},
    {"name": "Couch",    "priority": "high"},
    {"name": "Door",     "priority": "medium"},
    {"name": "Bed",      "priority": "medium"},
    {"name": "Sink",     "priority": "medium"},
    {"name": "Microwave","priority": "medium"},
    {"name": "Laptop",   "priority": "low"},
    {"name": "Bottle",   "priority": "low"},
    {"name": "Backpack", "priority": "low"},
    {"name": "Cup",      "priority": "low"},
]

DIRECTIONS = ["Left", "Ahead", "Right"]


def get_detections() -> list[dict]:
    """
    Returns a list of currently detected objects.

    ── REPLACE WITH YOUR REAL DATA ──────────────────────────────
    Connect this to your YOLOv8 output. Example real implementation:
        results = model.predict(frame)
        detections = []
        for box in results[0].boxes:
            detections.append({
                "name":       results[0].names[int(box.cls)],
                "confidence": float(box.conf),
                "direction":  get_direction(box, frame_width),
                "distance_cm": get_ultrasonic_distance(),
                "priority":   get_priority(results[0].names[int(box.cls)]),
            })
        return detections
    ─────────────────────────────────────────────────────────────
    """
    count = random.randint(1, 4)
    chosen = random.sample(OBJECT_POOL, min(count, len(OBJECT_POOL)))
    detections = []
    base_dist = random.randint(30, 200)
    for i, obj in enumerate(chosen):
        detections.append({
            "name":        obj["name"],
            "confidence":  round(random.uniform(0.72, 0.99), 2),
            "direction":   random.choice(DIRECTIONS),
            "distance_cm": base_dist + i * random.randint(5, 40),
            "priority":    obj["priority"],
        })
    return detections


def get_ultrasonic_distance() -> int:
    """
    Returns distance in cm from the ultrasonic sensor.

    ── REPLACE WITH YOUR REAL DATA ──────────────────────────────
    Example for RPi with HC-SR04 sensor:
        import RPi.GPIO as GPIO
        import time
        GPIO.setmode(GPIO.BCM)
        TRIG, ECHO = 23, 24
        GPIO.setup(TRIG, GPIO.OUT)
        GPIO.setup(ECHO, GPIO.IN)
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        while GPIO.input(ECHO) == 0: pulse_start = time.time()
        while GPIO.input(ECHO) == 1: pulse_end   = time.time()
        distance = round((pulse_end - pulse_start) * 17150, 1)
        return distance
    ─────────────────────────────────────────────────────────────
    """
    return random.randint(15, 295)


def get_system_stats() -> dict:
    """
    Returns system health metrics from the Raspberry Pi.

    ── REPLACE WITH YOUR REAL DATA ──────────────────────────────
    Example using psutil on Raspberry Pi:
        import psutil, subprocess
        cpu  = psutil.cpu_percent(interval=0.5)
        ram  = psutil.virtual_memory().percent
        temp_raw = subprocess.run(
            ['vcgencmd', 'measure_temp'],
            capture_output=True, text=True
        ).stdout
        temp = float(temp_raw.replace("temp=","").replace("'C\n",""))
        return {"cpu": cpu, "ram": ram, "temp": temp, "fps": 10}
    ─────────────────────────────────────────────────────────────
    """
    return {
        "cpu":  random.randint(30, 80),
        "ram":  random.randint(50, 85),
        "temp": random.randint(45, 72),
        "fps":  random.randint(8, 15),
    }


# ─────────────────────────────────────────────────────────────
#  SESSION STATE — alert log + uptime
# ─────────────────────────────────────────────────────────────
if "alert_log" not in st.session_state:
    st.session_state.alert_log = [
        {"time": "00:05", "msg": "Chair detected ahead — 45 cm",      "level": "danger"},
        {"time": "00:04", "msg": "Person detected on your left — 110 cm","level": "warn"},
        {"time": "00:02", "msg": "Table detected ahead — 88 cm",       "level": "warn"},
        {"time": "00:01", "msg": "Clear path",                         "level": "ok"},
    ]

if "alert_count" not in st.session_state:
    st.session_state.alert_count = 4

if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.datetime.now()

if "dist_history" not in st.session_state:
    st.session_state.dist_history = [random.randint(30, 250) for _ in range(20)]


def add_alert(msg: str, level: str):
    now = datetime.datetime.now().strftime("%M:%S")
    st.session_state.alert_log.insert(0, {"time": now, "msg": msg, "level": level})
    if len(st.session_state.alert_log) > 8:
        st.session_state.alert_log.pop()
    st.session_state.alert_count += 1


# ─────────────────────────────────────────────────────────────
#  FETCH LIVE DATA
# ─────────────────────────────────────────────────────────────
detections  = get_detections()
ultrasonic  = get_ultrasonic_distance()
sys_stats   = get_system_stats()
uptime      = datetime.datetime.now() - st.session_state.start_time
uptime_str  = str(uptime).split(".")[0]

# Update distance history
st.session_state.dist_history.append(ultrasonic)
if len(st.session_state.dist_history) > 30:
    st.session_state.dist_history.pop(0)

# Auto-generate alerts from detections
nearest = detections[0] if detections else None
if nearest:
    if nearest["priority"] == "high" and nearest["distance_cm"] < 80:
        add_alert(
            f"{nearest['name']} detected {nearest['direction'].lower()} — {nearest['distance_cm']} cm",
            "danger"
        )
    elif nearest["distance_cm"] < 150:
        add_alert(
            f"{nearest['name']} detected {nearest['direction'].lower()} — {nearest['distance_cm']} cm",
            "warn"
        )

avg_conf     = round(sum(d["confidence"] for d in detections) / len(detections) * 100)
nearest_dist = min(d["distance_cm"] for d in detections) if detections else 0


# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
col_title, col_time, col_refresh = st.columns([3, 2, 1])
with col_title:
    st.markdown("## 👁 VisualAid Detection Dashboard")
with col_time:
    st.markdown(
        f"<div style='padding-top:10px;color:#8b8fa8;font-size:13px'>"
        f"Last updated: {datetime.datetime.now().strftime('%H:%M:%S')} &nbsp;|&nbsp; "
        f"Uptime: {uptime_str}</div>",
        unsafe_allow_html=True
    )
with col_refresh:
    refresh = st.button("⟳ Refresh", use_container_width=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────
#  TOP METRICS
# ─────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Objects detected",  len(detections),                       delta=None)
m2.metric("Nearest object",    f"{nearest_dist} cm",                  delta=None)
m3.metric("Avg confidence",    f"{avg_conf}%",                        delta=None)
m4.metric("Alerts today",      st.session_state.alert_count,          delta=None)
m5.metric("Detection speed",   f"{sys_stats['fps']} fps",             delta=None)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  MAIN CONTENT — left panel | right panel
# ─────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")


# ── LEFT: Live detections ─────────────────────────────────────
with left:
    st.markdown('<div class="section-header">Live detections</div>', unsafe_allow_html=True)

    for det in detections:
        conf_pct = int(det["confidence"] * 100)
        badge_cls = f"badge-{det['priority']}"
        conf_color = (
            "#4ade80" if conf_pct >= 85
            else "#fbbf24" if conf_pct >= 70
            else "#f87171"
        )
        dist_color = (
            "#f87171" if det["distance_cm"] < 60
            else "#fbbf24" if det["distance_cm"] < 130
            else "#4ade80"
        )

        st.markdown(f"""
        <div class="det-card">
            <div>
                <div class="det-name">{det['name']}
                    &nbsp;<span class="badge {badge_cls}">{det['priority']}</span>
                </div>
                <div class="det-dir">{det['direction']}</div>
            </div>
            <div style="flex:1;margin:0 16px">
                <div style="height:4px;background:#2a2d3a;border-radius:2px;overflow:hidden">
                    <div style="width:{conf_pct}%;height:100%;background:{conf_color};border-radius:2px"></div>
                </div>
                <div style="font-size:11px;color:#8b8fa8;margin-top:4px">{conf_pct}% confidence</div>
            </div>
            <div class="det-dist" style="color:{dist_color}">{det['distance_cm']} cm</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Distance history chart ────────────────────────────────
    st.markdown('<div class="section-header">Ultrasonic distance — last 30 readings</div>',
                unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=st.session_state.dist_history,
        mode="lines+markers",
        line=dict(color="#7c6af7", width=2),
        marker=dict(size=4, color="#7c6af7"),
        fill="tozeroy",
        fillcolor="rgba(124,106,247,0.08)",
    ))
    fig.add_hline(y=60,  line_dash="dash", line_color="#ef4444",
                  annotation_text="Danger < 60 cm",
                  annotation_font_color="#ef4444", annotation_font_size=11)
    fig.add_hline(y=130, line_dash="dash", line_color="#f59e0b",
                  annotation_text="Caution < 130 cm",
                  annotation_font_color="#f59e0b", annotation_font_size=11)
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#1e2130", color="#8b8fa8",
                   range=[0, 310], ticksuffix=" cm", tickfont=dict(size=11)),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── RIGHT: Ultrasonic + Direction + Alert log + System ────────
with right:

    # Direction indicator
    st.markdown('<div class="section-header">Direction map</div>', unsafe_allow_html=True)

    nearest_dir = detections[0]["direction"] if detections else None
    nearest_pri = detections[0]["priority"]  if detections else None
    nearest_d   = detections[0]["distance_cm"] if detections else 999

    def dir_class(label):
        if nearest_dir != label:
            return "dir-cell"
        if nearest_pri == "high" or nearest_d < 60:
            return "dir-cell active-danger"
        return "dir-cell active-warn"

    st.markdown(f"""
    <div class="dir-grid">
        <div class="{dir_class('Left')}">◂ Left</div>
        <div class="{dir_class('Ahead')}">▲ Ahead</div>
        <div class="{dir_class('Right')}">Right ▸</div>
    </div>
    """, unsafe_allow_html=True)

    # Ultrasonic reading
    dist_color = (
        "#f87171" if ultrasonic < 60
        else "#fbbf24" if ultrasonic < 130
        else "#4ade80"
    )
    dist_label = (
        "DANGER — too close" if ultrasonic < 60
        else "CAUTION — nearby" if ultrasonic < 130
        else "CLEAR"
    )
    dist_pct = min(int((ultrasonic / 300) * 100), 100)

    st.markdown(f"""
    <div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;padding:14px 16px;margin:10px 0">
        <div class="section-header" style="margin-bottom:6px">Ultrasonic sensor</div>
        <div style="display:flex;align-items:baseline;gap:6px">
            <span style="font-size:32px;font-weight:700;color:{dist_color}">{ultrasonic}</span>
            <span style="color:#8b8fa8;font-size:14px">cm</span>
            <span style="margin-left:8px;font-size:11px;font-weight:700;color:{dist_color}">{dist_label}</span>
        </div>
        <div style="height:6px;background:#2a2d3a;border-radius:3px;overflow:hidden;margin-top:10px">
            <div style="width:{dist_pct}%;height:100%;background:{dist_color};border-radius:3px;transition:width .3s"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:4px;font-size:11px;color:#8b8fa8">
            <span>0 cm</span><span>300 cm</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Alert log
    st.markdown('<div class="section-header">Alert log</div>', unsafe_allow_html=True)
    for alert in st.session_state.alert_log[:6]:
        cls = {
            "danger": "alert-danger",
            "warn":   "alert-warn",
            "ok":     "alert-ok",
        }.get(alert["level"], "alert-ok")
        st.markdown(f"""
        <div class="alert-item {cls}">
            <span class="alert-time">{alert['time']}</span>
            <span class="alert-msg">{alert['msg']}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # System status
    st.markdown('<div class="section-header">System status</div>', unsafe_allow_html=True)

    cpu_color  = "status-warn" if sys_stats["cpu"]  > 70 else "status-on"
    ram_color  = "status-warn" if sys_stats["ram"]  > 75 else "status-on"
    temp_color = "status-warn" if sys_stats["temp"] > 65 else "status-on"

    st.markdown(f"""
    <div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;padding:10px 16px">
        <div class="sys-row">
            <span class="sys-label">Camera</span>
            <span class="status-on">Active</span>
        </div>
        <div class="sys-row">
            <span class="sys-label">YOLOv8 model</span>
            <span class="status-on">Running</span>
        </div>
        <div class="sys-row">
            <span class="sys-label">Ultrasonic sensor</span>
            <span class="status-on">Active</span>
        </div>
        <div class="sys-row">
            <span class="sys-label">Audio output</span>
            <span class="status-on">On</span>
        </div>
        <div class="sys-row">
            <span class="sys-label">CPU</span>
            <span class="{cpu_color}">{sys_stats['cpu']}%</span>
        </div>
        <div class="sys-row">
            <span class="sys-label">RAM</span>
            <span class="{ram_color}">{sys_stats['ram']}%</span>
        </div>
        <div class="sys-row">
            <span class="sys-label">Temperature</span>
            <span class="{temp_color}">{sys_stats['temp']}°C</span>
        </div>
        <div class="sys-row" style="border-bottom:none">
            <span class="sys-label">Uptime</span>
            <span class="sys-val" style="color:#e8eaf0">{uptime_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  DETECTION HISTORY TABLE
# ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Current detection details</div>',
            unsafe_allow_html=True)

if detections:
    df = pd.DataFrame(detections)
    df["confidence"] = (df["confidence"] * 100).round(1).astype(str) + "%"
    df["distance_cm"] = df["distance_cm"].astype(str) + " cm"
    df.columns = ["Object", "Confidence", "Direction", "Distance", "Priority"]
    df = df[["Object", "Priority", "Direction", "Distance", "Confidence"]]
    st.dataframe(df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
#  AUTO REFRESH every 2 seconds
# ─────────────────────────────────────────────────────────────
time.sleep(2)
st.rerun()
