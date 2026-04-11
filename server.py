# server.py  —  Web dashboard server
#
# Routes:
#   /        → dashboard HTML
#   /video   → MJPEG stream
#   /state   → JSON state (polled by dashboard every 300ms)
#
# Run via run.py — do not run this file directly.

import logging
import time
from flask import Flask, Response, jsonify
import shared_state

log = logging.getLogger("server")
app = Flask(__name__)


# ── MJPEG stream ──────────────────────────────────────────────────────────────

def _frames():
    while True:
        jpeg = shared_state.latest_jpeg()
        if jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                b"\r\n" +
                jpeg +
                b"\r\n"
            )
        time.sleep(0.05)   # 20fps cap — enough for a demo, light on Pi CPU


@app.route("/video")
def video():
    return Response(
        _frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma":        "no-cache",
            "Expires":       "0",
        }
    )


# ── State ─────────────────────────────────────────────────────────────────────

@app.route("/state")
def state():
    r = jsonify(shared_state.get_state())
    r.headers["Cache-Control"] = "no-cache"
    return r


# ── Health check (useful for debugging) ──────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": time.time()})


# ── Dashboard ─────────────────────────────────────────────────────────────────

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
    --bg:       #080c10;
    --surface:  #0d1117;
    --surface2: #111820;
    --border:   #1e2d3d;
    --accent:   #00d4ff;
    --green:    #0aff9d;
    --urgent:   #ff4d4d;
    --warning:  #ffaa00;
    --txt:      #cdd9e5;
    --muted:    #4a5568;
    --mono:     'Space Mono', monospace;
    --sans:     'Syne', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    background: var(--bg);
    color: var(--txt);
    font-family: var(--sans);
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  /* Header */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 24px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .logo { display: flex; align-items: center; gap: 10px; }
  .dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
    animation: blink 2s ease-in-out infinite;
  }
  @keyframes blink {
    0%,100% { opacity:1; box-shadow: 0 0 8px var(--green); }
    50%      { opacity:.4; box-shadow: 0 0 16px var(--green); }
  }
  .logo h1 { font-size:17px; font-weight:800; letter-spacing:.07em; color:#fff; }
  .logo h1 span { color: var(--accent); }
  .hdr-right { display:flex; align-items:center; gap:16px; }
  #uptime { font-family:var(--mono); font-size:11px; color:var(--muted); }
  .live {
    background: rgba(255,77,77,.12);
    border: 1px solid rgba(255,77,77,.35);
    color: #ff6b6b;
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: .12em;
    padding: 3px 9px;
    border-radius: 3px;
  }

  /* Layout */
  .body {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 290px;
    gap: 1px;
    background: var(--border);
    overflow: hidden;
  }

  /* Camera */
  .cam-wrap {
    background: #000;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  .cam-wrap img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
  /* scanlines */
  .cam-wrap::after {
    content:'';
    position:absolute; inset:0;
    background: repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.04) 2px,rgba(0,0,0,.04) 4px);
    pointer-events: none;
  }
  /* Alert overlay */
  #banner {
    position: absolute;
    top: 0; left: 0; right: 0;
    padding: 10px 16px;
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: .04em;
    opacity: 0;
    transition: opacity .25s;
    pointer-events: none;
    z-index: 10;
  }
  #banner.show { opacity: 1; }
  #banner.p3 { background: rgba(255,77,77,.88); color:#fff; }
  #banner.p2 { background: rgba(255,170,0,.88);  color:#000; }
  #banner.p1 { background: rgba(0,212,255,.82);  color:#000; }

  /* Sidebar */
  .sidebar {
    background: var(--surface);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    gap: 1px;
  }
  .panel { background: var(--surface2); padding: 14px 16px; }
  .ptitle {
    font-size: 8px;
    font-weight: 600;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .ptitle::after { content:''; flex:1; height:1px; background:var(--border); }

  /* Zones */
  .zones { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; }
  .zone {
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px 4px;
    text-align: center;
    transition: all .3s;
  }
  .zname { font-size:8px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin-bottom:5px; }
  .zval  { font-family:var(--mono); font-size:9px; font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
  .zbar  { margin-top:5px; height:3px; border-radius:2px; background:var(--border); overflow:hidden; }
  .zbarf { height:100%; border-radius:2px; width:0; transition: width .4s, background .3s; }

  .s-clear    { color:var(--green);   border-color:rgba(10,255,157,.22); background:rgba(10,255,157,.04); }
  .s-occupied { color:var(--warning); border-color:rgba(255,170,0,.28);  background:rgba(255,170,0,.05); }
  .s-crowded  { color:var(--urgent);  border-color:rgba(255,77,77,.28);  background:rgba(255,77,77,.05); }

  /* Sensor */
  .sen-row { display:flex; align-items:flex-end; gap:5px; margin-bottom:6px; }
  #sen-cm  { font-family:var(--mono); font-size:34px; font-weight:700; color:var(--accent); transition:color .3s; line-height:1; }
  .sen-unit { font-family:var(--mono); font-size:11px; color:var(--muted); margin-bottom:3px; }
  #sen-band {
    display:inline-block;
    font-family:var(--mono); font-size:9px; letter-spacing:.1em;
    text-transform:uppercase; padding:2px 7px; border-radius:3px; margin-bottom:4px;
  }
  .bc { background:rgba(255,77,77,.18); color:#ff4d4d; }
  .bw { background:rgba(255,140,0,.18); color:#ff8c00; }
  .bm { background:rgba(255,170,0,.14); color:var(--warning); }
  .bf { background:rgba(10,255,157,.1); color:var(--green); }
  .bn { background:rgba(100,100,100,.1); color:var(--muted); }
  #sen-src { font-family:var(--mono); font-size:9px; color:var(--muted); }
  #floor-warn {
    display:none; margin-top:7px; padding:5px 9px;
    background:rgba(255,77,77,.1); border:1px solid rgba(255,77,77,.28);
    border-radius:3px; font-family:var(--mono); font-size:9px; color:#ff6b6b;
    letter-spacing:.05em;
  }

  /* Metrics */
  .metrics { display:grid; grid-template-columns:1fr 1fr; gap:5px; }
  .metric {
    background:var(--surface); border:1px solid var(--border);
    border-radius:4px; padding:9px;
  }
  .mlabel { font-size:8px; letter-spacing:.13em; text-transform:uppercase; color:var(--muted); margin-bottom:3px; }
  .mval   { font-family:var(--mono); font-size:20px; font-weight:700; color:#fff; line-height:1; }

  /* Log */
  .log-wrap { flex:1; overflow-y:auto; display:flex; flex-direction:column; gap:3px; max-height:220px;
    scrollbar-width:thin; scrollbar-color:var(--border) transparent; }
  .log-entry {
    display:flex; gap:7px; align-items:flex-start;
    padding:5px 7px; border-radius:3px;
    border-left:2px solid var(--border);
    background:var(--surface);
    animation: li .2s ease;
  }
  @keyframes li { from{opacity:0;transform:translateX(-5px)} to{opacity:1;transform:none} }
  .lp3 { border-left-color:var(--urgent);  background:rgba(255,77,77,.06); }
  .lp2 { border-left-color:var(--warning); background:rgba(255,170,0,.05); }
  .lp1 { border-left-color:var(--accent);  background:rgba(0,212,255,.04); }
  .ltime { font-family:var(--mono); font-size:9px; color:var(--muted); white-space:nowrap; flex-shrink:0; padding-top:1px; }
  .lmsg  { font-family:var(--mono); font-size:10px; line-height:1.4; }

  /* Footer */
  footer {
    border-top:1px solid var(--border); background:var(--surface);
    padding:6px 24px; display:flex; justify-content:space-between;
    flex-shrink:0;
  }
  footer span { font-family:var(--mono); font-size:9px; color:var(--muted); letter-spacing:.06em; }
  #poll-rate { color:var(--green); }

  ::-webkit-scrollbar { width:3px; }
  ::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="dot"></div>
    <h1>Vision<span>Guide</span></h1>
  </div>
  <div class="hdr-right">
    <span id="uptime">UP 00:00:00</span>
    <span class="live">● LIVE</span>
  </div>
</header>

<div class="body">
  <div class="cam-wrap">
    <img id="stream" src="/video" alt="camera feed">
    <div id="banner"></div>
  </div>

  <div class="sidebar">

    <div class="panel">
      <div class="ptitle">Navigation Zones</div>
      <div class="zones" id="zones">
        <div class="zone s-clear" id="zl">
          <div class="zname">Left</div>
          <div class="zval" id="zl-v">clear</div>
          <div class="zbar"><div class="zbarf" id="zl-b"></div></div>
        </div>
        <div class="zone s-clear" id="zc">
          <div class="zname">Center</div>
          <div class="zval" id="zc-v">clear</div>
          <div class="zbar"><div class="zbarf" id="zc-b"></div></div>
        </div>
        <div class="zone s-clear" id="zr">
          <div class="zname">Right</div>
          <div class="zval" id="zr-v">clear</div>
          <div class="zbar"><div class="zbarf" id="zr-b"></div></div>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="ptitle">Ultrasonic Sensor</div>
      <div class="sen-row">
        <div id="sen-cm">—</div>
        <div class="sen-unit">cm</div>
      </div>
      <div id="sen-band" class="bn">NO READING</div>
      <div id="sen-src" style="margin-top:4px">source: —</div>
      <div id="floor-warn">⚠ UNSEEN OBSTACLE</div>
    </div>

    <div class="panel">
      <div class="ptitle">Scene Metrics</div>
      <div class="metrics">
        <div class="metric"><div class="mlabel">Objects</div><div class="mval" id="m-obj">0</div></div>
        <div class="metric"><div class="mlabel">Confirmed</div><div class="mval" id="m-con">0</div></div>
        <div class="metric"><div class="mlabel">Closest</div><div class="mval" id="m-prx" style="font-size:13px;padding-top:3px">—</div></div>
        <div class="metric"><div class="mlabel">Alerts</div><div class="mval" id="m-alt">0</div></div>
      </div>
    </div>

    <div class="panel" style="flex:1;display:flex;flex-direction:column;">
      <div class="ptitle">Alert Log</div>
      <div class="log-wrap" id="log"></div>
    </div>

  </div>
</div>

<footer>
  <span>VisionGuide Navigation Aid</span>
  <span id="poll-rate">connecting...</span>
</footer>

<script>
  const SC = { clear:'s-clear', occupied:'s-occupied', crowded:'s-crowded' };
  const BC = { critical:'bc', close:'bw', medium:'bm', far:'bf', none:'bn' };
  const ZONE_COLS = { clear:'var(--green)', occupied:'var(--warning)', crowded:'var(--urgent)' };
  const PROX_COLS = { close:'var(--urgent)', medium:'var(--warning)', far:'var(--green)', none:'var(--muted)' };
  const SENS_COLS = { critical:'#ff4d4d', close:'#ff8c00', medium:'var(--warning)', far:'var(--green)', none:'var(--muted)' };

  let lastAlerts = 0, bannerTmo = null;
  let pc = 0, ps = Date.now();

  const $ = id => document.getElementById(id);

  function fmt(s) {
    const h=String(Math.floor(s/3600)).padStart(2,'0');
    const m=String(Math.floor(s%3600/60)).padStart(2,'0');
    const sec=String(s%60).padStart(2,'0');
    return `UP ${h}:${m}:${sec}`;
  }

  function ftime() {
    return new Date().toLocaleTimeString('en-GB',{hour12:false});
  }

  function setZone(id, status, count) {
    const el = $(id), vl = $(`${id}-v`), br = $(`${id}-b`);
    el.className = `zone ${SC[status]||'s-clear'}`;
    vl.textContent = status;
    const pct = Math.min(count/3,1)*100;
    br.style.width = pct>0 ? pct+'%' : '0%';
    br.style.background = ZONE_COLS[status]||'var(--border)';
  }

  function showBanner(msg, pri) {
    const b = $('banner');
    b.textContent = msg;
    b.className = `show p${pri}`;
    if (bannerTmo) clearTimeout(bannerTmo);
    bannerTmo = setTimeout(()=>{ b.className=''; }, 4000);
  }

  function addLog(msg, pri) {
    const l = $('log');
    const e = document.createElement('div');
    e.className = `log-entry lp${pri}`;
    e.innerHTML = `<span class="ltime">${ftime()}</span><span class="lmsg">${msg}</span>`;
    l.insertBefore(e, l.firstChild);
    while(l.children.length > 50) l.removeChild(l.lastChild);
  }

  // Reconnecting stream — if image errors, retry after 2s
  const img = $('stream');
  img.onerror = () => {
    setTimeout(() => { img.src = '/video?' + Date.now(); }, 2000);
  };

  async function poll() {
    try {
      const s = await (await fetch('/state')).json();

      $('uptime').textContent = fmt(s.uptime_seconds||0);

      setZone('zl', s.zones.left,   s.zone_counts.left);
      setZone('zc', s.zones.center, s.zone_counts.center);
      setZone('zr', s.zones.right,  s.zone_counts.right);

      const band = s.sensor_band||'none';
      const cm   = s.sensor_cm;
      $('sen-cm').textContent  = cm !== null ? Math.round(cm) : '—';
      $('sen-cm').style.color  = SENS_COLS[band]||'var(--accent)';
      $('sen-band').textContent = band.toUpperCase();
      $('sen-band').className  = BC[band]||'bn';
      $('sen-src').textContent = `source: ${s.sensor_source||'—'}`;
      $('floor-warn').style.display = s.sensor_floor ? 'block' : 'none';

      $('m-obj').textContent = s.object_count||0;
      $('m-con').textContent = s.confirmed_count||0;
      $('m-alt').textContent = s.alert_count||0;
      const prx = s.closest_proximity||'none';
      $('m-prx').textContent  = prx !== 'none' ? prx : '—';
      $('m-prx').style.color  = PROX_COLS[prx]||'var(--txt)';

      if ((s.alert_count||0) > lastAlerts && s.last_message) {
        showBanner(s.last_message, s.last_priority||1);
        addLog(s.last_message, s.last_priority||1);
        lastAlerts = s.alert_count;
      }

      pc++;
      const el = (Date.now()-ps)/1000;
      if (el >= 2) {
        $('poll-rate').textContent = `${(pc/el).toFixed(1)} polls/s`;
        pc=0; ps=Date.now();
      }
    } catch(e) {
      $('poll-rate').textContent = 'reconnecting...';
    }
    setTimeout(poll, 300);
  }

  poll();
</script>
</body>
</html>"""


@app.route("/")
def dashboard():
    return DASHBOARD_HTML


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)-10s %(message)s")
    log.info("Starting server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True,
            debug=False, use_reloader=False)