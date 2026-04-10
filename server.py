# server.py  —  Web dashboard server
#
# Serves the live browser dashboard on port 5000.
# Reads frames and state from shared_state.py (populated by main.py).
#
# Routes:
#   /          → dashboard HTML page
#   /video     → MJPEG video stream (camera feed)
#   /state     → JSON state snapshot (polled every second by the page)

import logging
import time
from flask import Flask, Response, jsonify
import shared_state

log = logging.getLogger("server")
app = Flask(__name__)


# ── Video stream ──────────────────────────────────────────────────────────────

def _generate_frames():
    """Yield MJPEG frames from shared_state indefinitely."""
    while True:
        jpeg = shared_state.latest_jpeg()
        if jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg +
                b"\r\n"
            )
        time.sleep(0.033)   # ~30 fps cap — browser renders as fast as Pi produces


@app.route("/video")
def video():
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ── State JSON ────────────────────────────────────────────────────────────────

@app.route("/state")
def state():
    return jsonify(shared_state.get_state())


# ── Dashboard HTML ────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VisionGuide — Live Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #080c10;
    --surface:   #0d1117;
    --surface2:  #131920;
    --border:    #1e2d3d;
    --accent:    #00d4ff;
    --accent2:   #0aff9d;
    --urgent:    #ff4d4d;
    --warning:   #ffaa00;
    --clear:     #00d4ff;
    --txt:       #cdd9e5;
    --txt-muted: #4a5568;
    --mono:      'Space Mono', monospace;
    --sans:      'Syne', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--txt);
    font-family: var(--sans);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── Header ── */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .logo-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--accent2);
    box-shadow: 0 0 10px var(--accent2);
    animation: pulse 2s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--accent2); }
    50%       { opacity: 0.5; box-shadow: 0 0 20px var(--accent2); }
  }

  .logo h1 {
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #fff;
  }

  .logo span { color: var(--accent); }

  .header-right {
    display: flex;
    align-items: center;
    gap: 20px;
  }

  #uptime {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--txt-muted);
    letter-spacing: 0.05em;
  }

  .live-badge {
    background: rgba(255,77,77,0.15);
    border: 1px solid rgba(255,77,77,0.4);
    color: #ff6b6b;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    padding: 3px 10px;
    border-radius: 3px;
    text-transform: uppercase;
  }

  /* ── Main layout ── */
  .main {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 300px;
    gap: 1px;
    background: var(--border);
    overflow: hidden;
  }

  /* ── Camera panel ── */
  .camera-panel {
    background: #000;
    display: flex;
    flex-direction: column;
    position: relative;
  }

  .camera-panel img {
    width: 100%;
    display: block;
    flex: 1;
    object-fit: contain;
    background: #000;
  }

  /* Alert banner over video */
  .alert-banner {
    position: absolute;
    top: 0; left: 0; right: 0;
    padding: 10px 18px;
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.04em;
    transition: background 0.3s, color 0.3s, opacity 0.5s;
    opacity: 0;
    pointer-events: none;
  }

  .alert-banner.visible { opacity: 1; }
  .alert-banner.urgent  { background: rgba(255,77,77,0.85);  color: #fff; }
  .alert-banner.warning { background: rgba(255,170,0,0.85);  color: #000; }
  .alert-banner.clear   { background: rgba(0,212,255,0.85);  color: #000; }

  /* Scanline overlay — subtle CRT feel */
  .camera-panel::after {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,0,0,0.03) 2px,
      rgba(0,0,0,0.03) 4px
    );
    pointer-events: none;
  }

  /* ── Sidebar ── */
  .sidebar {
    background: var(--surface);
    display: flex;
    flex-direction: column;
    gap: 1px;
    overflow-y: auto;
  }

  .panel {
    background: var(--surface2);
    padding: 16px;
  }

  .panel-title {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--txt-muted);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .panel-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* ── Zone grid ── */
  .zones {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 6px;
  }

  .zone {
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px 6px;
    text-align: center;
    transition: border-color 0.3s, background 0.3s;
  }

  .zone-name {
    font-size: 8px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--txt-muted);
    margin-bottom: 6px;
  }

  .zone-status {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .zone-bar {
    margin-top: 6px;
    height: 3px;
    border-radius: 2px;
    background: var(--border);
    overflow: hidden;
  }

  .zone-bar-fill {
    height: 100%;
    border-radius: 2px;
    width: 0%;
    transition: width 0.4s ease, background 0.3s;
  }

  .status-clear    { color: var(--accent2); border-color: rgba(10,255,157,0.2); background: rgba(10,255,157,0.04); }
  .status-occupied { color: var(--warning); border-color: rgba(255,170,0,0.3);  background: rgba(255,170,0,0.05); }
  .status-crowded  { color: var(--urgent);  border-color: rgba(255,77,77,0.3);  background: rgba(255,77,77,0.05); }

  /* ── Sensor panel ── */
  .sensor-reading {
    display: flex;
    align-items: flex-end;
    gap: 6px;
    margin-bottom: 8px;
  }

  .sensor-cm {
    font-family: var(--mono);
    font-size: 36px;
    font-weight: 700;
    line-height: 1;
    color: var(--accent);
    transition: color 0.3s;
  }

  .sensor-unit {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--txt-muted);
    margin-bottom: 4px;
  }

  .sensor-band {
    display: inline-block;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 3px;
    margin-bottom: 4px;
  }

  .band-critical { background: rgba(255,77,77,0.2);   color: #ff4d4d; }
  .band-close    { background: rgba(255,140,0,0.2);   color: #ff8c00; }
  .band-medium   { background: rgba(255,170,0,0.15);  color: #ffaa00; }
  .band-far      { background: rgba(10,255,157,0.1);  color: var(--accent2); }
  .band-none     { background: rgba(100,100,100,0.1); color: var(--txt-muted); }

  .sensor-source {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--txt-muted);
    letter-spacing: 0.06em;
  }

  .floor-warning {
    margin-top: 8px;
    padding: 6px 10px;
    background: rgba(255,77,77,0.1);
    border: 1px solid rgba(255,77,77,0.3);
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 10px;
    color: #ff6b6b;
    letter-spacing: 0.06em;
    display: none;
  }

  /* ── Metrics ── */
  .metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
  }

  .metric {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px;
  }

  .metric-label {
    font-size: 8px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--txt-muted);
    margin-bottom: 4px;
  }

  .metric-value {
    font-family: var(--mono);
    font-size: 22px;
    font-weight: 700;
    color: #fff;
    line-height: 1;
  }

  /* ── Alert log ── */
  .log-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 240px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
  }

  .log-entry {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    padding: 6px 8px;
    border-radius: 3px;
    background: var(--surface);
    border-left: 2px solid var(--border);
    animation: slideIn 0.2s ease;
  }

  @keyframes slideIn {
    from { opacity: 0; transform: translateX(-6px); }
    to   { opacity: 1; transform: translateX(0); }
  }

  .log-entry.urgent  { border-left-color: var(--urgent);  background: rgba(255,77,77,0.06); }
  .log-entry.warning { border-left-color: var(--warning); background: rgba(255,170,0,0.06); }
  .log-entry.clear   { border-left-color: var(--accent2); background: rgba(10,255,157,0.04); }

  .log-time {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--txt-muted);
    white-space: nowrap;
    flex-shrink: 0;
    padding-top: 1px;
  }

  .log-msg {
    font-family: var(--mono);
    font-size: 10px;
    line-height: 1.4;
    color: var(--txt);
  }

  /* ── Footer ── */
  footer {
    border-top: 1px solid var(--border);
    background: var(--surface);
    padding: 8px 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  footer span {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--txt-muted);
    letter-spacing: 0.06em;
  }

  #fps-counter { color: var(--accent2); }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-dot"></div>
    <h1>Vision<span>Guide</span></h1>
  </div>
  <div class="header-right">
    <span id="uptime">UP 00:00:00</span>
    <span class="live-badge">● Live</span>
  </div>
</header>

<div class="main">

  <!-- Camera feed -->
  <div class="camera-panel">
    <img src="/video" alt="Live camera feed">
    <div class="alert-banner" id="alert-banner"></div>
  </div>

  <!-- Sidebar -->
  <div class="sidebar">

    <!-- Zones -->
    <div class="panel">
      <div class="panel-title">Navigation Zones</div>
      <div class="zones">
        <div class="zone status-clear" id="zone-left">
          <div class="zone-name">Left</div>
          <div class="zone-status" id="zone-left-status">clear</div>
          <div class="zone-bar"><div class="zone-bar-fill" id="zone-left-bar"></div></div>
        </div>
        <div class="zone status-clear" id="zone-center">
          <div class="zone-name">Center</div>
          <div class="zone-status" id="zone-center-status">clear</div>
          <div class="zone-bar"><div class="zone-bar-fill" id="zone-center-bar"></div></div>
        </div>
        <div class="zone status-clear" id="zone-right">
          <div class="zone-name">Right</div>
          <div class="zone-status" id="zone-right-status">clear</div>
          <div class="zone-bar"><div class="zone-bar-fill" id="zone-right-bar"></div></div>
        </div>
      </div>
    </div>

    <!-- Sensor -->
    <div class="panel">
      <div class="panel-title">Ultrasonic Sensor</div>
      <div class="sensor-reading">
        <div class="sensor-cm" id="sensor-cm">—</div>
        <div class="sensor-unit">cm</div>
      </div>
      <div class="sensor-band band-none" id="sensor-band">no reading</div>
      <div class="sensor-source" id="sensor-source">source: —</div>
      <div class="floor-warning" id="floor-warning">⚠ UNSEEN OBSTACLE DETECTED</div>
    </div>

    <!-- Metrics -->
    <div class="panel">
      <div class="panel-title">Scene Metrics</div>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">Objects</div>
          <div class="metric-value" id="m-objects">0</div>
        </div>
        <div class="metric">
          <div class="metric-label">Confirmed</div>
          <div class="metric-value" id="m-confirmed">0</div>
        </div>
        <div class="metric">
          <div class="metric-label">Closest</div>
          <div class="metric-value" id="m-closest" style="font-size:14px;padding-top:4px">—</div>
        </div>
        <div class="metric">
          <div class="metric-label">Alerts</div>
          <div class="metric-value" id="m-alerts">0</div>
        </div>
      </div>
    </div>

    <!-- Alert log -->
    <div class="panel" style="flex:1">
      <div class="panel-title">Alert Log</div>
      <div class="log-list" id="log-list"></div>
    </div>

  </div>
</div>

<footer>
  <span>VisionGuide Navigation Aid</span>
  <span id="fps-counter">polling...</span>
</footer>

<script>
  const ZONE_CLASSES = {
    clear:    'status-clear',
    occupied: 'status-occupied',
    crowded:  'status-crowded',
  };

  const BAND_CLASSES = {
    critical: 'band-critical',
    close:    'band-close',
    medium:   'band-medium',
    far:      'band-far',
    none:     'band-none',
  };

  let lastAlertCount = 0;
  let bannerTimer    = null;
  let pollCount      = 0;
  let pollStart      = Date.now();

  function fmtUptime(sec) {
    const h = String(Math.floor(sec / 3600)).padStart(2, '0');
    const m = String(Math.floor((sec % 3600) / 60)).padStart(2, '0');
    const s = String(sec % 60).padStart(2, '0');
    return `UP ${h}:${m}:${s}`;
  }

  function fmtTime() {
    return new Date().toLocaleTimeString('en-GB', { hour12: false });
  }

  function setZone(name, status, count) {
    const el  = document.getElementById(`zone-${name}`);
    const st  = document.getElementById(`zone-${name}-status`);
    const bar = document.getElementById(`zone-${name}-bar`);

    el.className = `zone ${ZONE_CLASSES[status] || 'status-clear'}`;
    st.textContent = status;

    const pct = Math.min(count / 3, 1) * 100;
    bar.style.width = pct > 0 ? `${pct}%` : '0%';

    const barColors = {
      clear: 'var(--accent2)', occupied: 'var(--warning)', crowded: 'var(--urgent)'
    };
    bar.style.background = barColors[status] || 'var(--border)';
  }

  function showBanner(msg, priority) {
    const banner = document.getElementById('alert-banner');
    const cls    = priority >= 3 ? 'urgent' : priority === 2 ? 'warning' : 'clear';
    banner.textContent = msg;
    banner.className   = `alert-banner visible ${cls}`;
    if (bannerTimer) clearTimeout(bannerTimer);
    bannerTimer = setTimeout(() => {
      banner.classList.remove('visible');
    }, 4000);
  }

  function addLog(msg, priority) {
    const list = document.getElementById('log-list');
    const cls  = priority >= 3 ? 'urgent' : priority === 2 ? 'warning' : 'clear';
    const entry = document.createElement('div');
    entry.className = `log-entry ${cls}`;
    entry.innerHTML = `
      <span class="log-time">${fmtTime()}</span>
      <span class="log-msg">${msg}</span>
    `;
    list.insertBefore(entry, list.firstChild);
    // Keep log to 40 entries
    while (list.children.length > 40) list.removeChild(list.lastChild);
  }

  async function poll() {
    try {
      const res   = await fetch('/state');
      const state = await res.json();

      // Uptime
      document.getElementById('uptime').textContent = fmtUptime(state.uptime_seconds);

      // Zones
      for (const zone of ['left', 'center', 'right']) {
        setZone(zone, state.zones[zone], state.zone_counts[zone]);
      }

      // Sensor
      const cm   = state.sensor_cm;
      const band = state.sensor_band || 'none';
      const cmEl = document.getElementById('sensor-cm');
      cmEl.textContent = cm !== null ? Math.round(cm) : '—';

      const sensorColors = {
        critical: '#ff4d4d', close: '#ff8c00',
        medium: '#ffaa00',   far: 'var(--accent2)', none: 'var(--txt-muted)'
      };
      cmEl.style.color = sensorColors[band] || 'var(--accent)';

      const bandEl = document.getElementById('sensor-band');
      bandEl.textContent = band.toUpperCase();
      bandEl.className   = `sensor-band ${BAND_CLASSES[band] || 'band-none'}`;

      document.getElementById('sensor-source').textContent =
        `source: ${state.sensor_source || '—'}`;

      const floorEl = document.getElementById('floor-warning');
      floorEl.style.display = state.sensor_floor ? 'block' : 'none';

      // Metrics
      document.getElementById('m-objects').textContent   = state.object_count;
      document.getElementById('m-confirmed').textContent = state.confirmed_count;
      document.getElementById('m-alerts').textContent    = state.alert_count;

      const prox = state.closest_proximity;
      const cls  = document.getElementById('m-closest');
      cls.textContent = prox !== 'none' ? prox : '—';
      cls.style.color = {
        close: 'var(--urgent)', medium: 'var(--warning)',
        far: 'var(--accent2)', none: 'var(--txt-muted)'
      }[prox] || 'var(--txt)';

      // New alert
      if (state.alert_count > lastAlertCount && state.last_message) {
        showBanner(state.last_message, state.last_priority);
        addLog(state.last_message, state.last_priority);
        lastAlertCount = state.alert_count;
      }

      // FPS counter (poll rate)
      pollCount++;
      const elapsed = (Date.now() - pollStart) / 1000;
      if (elapsed >= 2) {
        document.getElementById('fps-counter').textContent =
          `${(pollCount / elapsed).toFixed(1)} polls/s`;
        pollCount  = 0;
        pollStart  = Date.now();
      }

    } catch (e) {
      document.getElementById('fps-counter').textContent = 'connection lost';
    }

    setTimeout(poll, 250);   // poll 4x per second
  }

  // Start polling
  poll();
</script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return DASHBOARD_HTML


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-10s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("Dashboard server starting on http://0.0.0.0:5000")
    log.info("Open http://<pi-ip-address>:5000 in your browser")
    # threaded=True so the MJPEG stream and state polling don't block each other
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
