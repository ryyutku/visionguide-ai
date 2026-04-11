# cloud_logger.py
#
# Sends navigation events to Supabase in a background thread.
# The main inference loop is never blocked — events are queued
# and sent asynchronously.
#
# Setup (do once):
#   1. Create a free Supabase account at https://supabase.com
#   2. Create a new project (free tier is enough)
#   3. In the SQL editor, run the schema in supabase_schema.sql
#   4. Copy your project URL and anon key into .env  (see below)
#
# Environment variables (put in ~/visionguide/.env):
#   SUPABASE_URL=https://xxxx.supabase.co
#   SUPABASE_KEY=your-anon-key-here
#
# If these are not set, cloud logging is silently disabled —
# the rest of the system works normally without it.

import os
import threading
import queue
import time
import logging

log = logging.getLogger("cloud")

# Load .env file if present (simple parser, no python-dotenv needed)
def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_env()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
ENABLED      = bool(SUPABASE_URL and SUPABASE_KEY)


class CloudLogger:
    """
    Queues events and sends them to Supabase in a background thread.
    Call log_alert() and log_sensor() from the inference loop —
    they return immediately, never blocking YOLO.
    """

    # Max events to buffer if network is slow
    QUEUE_MAX = 200
    # Batch up to this many rows per HTTP request
    BATCH_SIZE = 10
    # How long to wait before flushing a partial batch
    FLUSH_INTERVAL = 2.0   # seconds

    def __init__(self, session_id: str = None):
        self._enabled    = ENABLED
        self._session_id = session_id or _make_session_id()
        self._q: queue.Queue = queue.Queue(maxsize=self.QUEUE_MAX)
        self._running    = True

        if not self._enabled:
            log.info("Cloud logging disabled (no SUPABASE_URL/KEY in .env)")
            return

        # Lazy import so missing 'requests' doesn't crash the whole app
        try:
            import requests
            self._requests = requests
        except ImportError:
            log.warning("pip install requests  — cloud logging disabled")
            self._enabled = False
            return

        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="cloud-logger"
        )
        self._thread.start()
        log.info("Cloud logging active  session=%s", self._session_id)

    # ── Public API ────────────────────────────────────────────────────────

    def log_alert(self, message: str, priority: int,
                  zone_states: dict, closest_class: str,
                  closest_region: str, closest_proximity: str):
        """Call when guidance.decide() returns a message."""
        if not self._enabled:
            return
        self._enqueue("alerts", {
            "session_id":        self._session_id,
            "message":           message,
            "priority":          priority,
            "zone_left":         zone_states.get("left",   "clear"),
            "zone_center":       zone_states.get("center", "clear"),
            "zone_right":        zone_states.get("right",  "clear"),
            "closest_class":     closest_class,
            "closest_region":    closest_region,
            "closest_proximity": closest_proximity,
        })

    def log_sensor(self, sensor_cm: float | None, sensor_band: str,
                   object_count: int, confirmed_count: int):
        """Call every N frames to track sensor + detection stats over time."""
        if not self._enabled:
            return
        self._enqueue("sensor_readings", {
            "session_id":      self._session_id,
            "sensor_cm":       sensor_cm,
            "sensor_band":     sensor_band,
            "object_count":    object_count,
            "confirmed_count": confirmed_count,
        })

    def shutdown(self):
        self._running = False

    # ── Internal ──────────────────────────────────────────────────────────

    def _enqueue(self, table: str, row: dict):
        row["ts"] = _iso_now()
        try:
            self._q.put_nowait({"table": table, "row": row})
        except queue.Full:
            pass   # drop silently — never block inference

    def _worker(self):
        batch_by_table: dict[str, list] = {}
        last_flush = time.time()

        while self._running:
            # Drain queue into local batches
            try:
                item = self._q.get(timeout=0.5)
                t    = item["table"]
                batch_by_table.setdefault(t, []).append(item["row"])
            except queue.Empty:
                pass

            # Flush when batch is big enough OR flush interval elapsed
            now = time.time()
            should_flush = (
                any(len(v) >= self.BATCH_SIZE for v in batch_by_table.values())
                or (now - last_flush >= self.FLUSH_INTERVAL and batch_by_table)
            )

            if should_flush:
                for table, rows in list(batch_by_table.items()):
                    if rows:
                        self._send(table, rows)
                        batch_by_table[table] = []
                batch_by_table = {k: v for k, v in batch_by_table.items() if v}
                last_flush = now

    def _send(self, table: str, rows: list):
        url     = f"{SUPABASE_URL}/rest/v1/{table}"
        headers = {
            "apikey":        SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type":  "application/json",
            "Prefer":        "return=minimal",
        }
        try:
            r = self._requests.post(url, json=rows, headers=headers, timeout=5)
            if r.status_code not in (200, 201):
                log.warning("Supabase %s: %d %s", table, r.status_code, r.text[:80])
        except Exception as e:
            log.debug("Cloud send failed: %s", e)


def _make_session_id() -> str:
    import uuid, datetime
    return (datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_" + str(uuid.uuid4())[:6])


def _iso_now() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"