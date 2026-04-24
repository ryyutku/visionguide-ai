# cloud_logger.py
#
# Sends navigation events to Supabase and polls for remote commands.
# Non-blocking — events are queued and sent in a background thread.
#
# Setup:
#   1. Create free Supabase account at https://supabase.com
#   2. Run supabase_schema.sql in the SQL editor
#   3. Create ~/visionguide/.env with:
#        SUPABASE_URL=https://xxxx.supabase.co
#        SUPABASE_KEY=your-anon-key-here
#        DEVICE_ID=pi_visionguide_01   (optional)
#
# If credentials are missing, cloud logging is silently disabled.

import os
import threading
import queue
import time
import logging
import socket

log = logging.getLogger("cloud")


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

# Exported at module level so run.py can import them for image uploads
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
DEVICE_ID    = os.environ.get("DEVICE_ID", socket.gethostname())
ENABLED      = bool(SUPABASE_URL and SUPABASE_KEY)


class CloudLogger:

    QUEUE_MAX        = 200
    BATCH_SIZE       = 10
    FLUSH_INTERVAL   = 2.0    # seconds
    COMMAND_INTERVAL = 5.0    # seconds between command polls

    def __init__(self, session_id: str = None):
        self._enabled    = ENABLED
        self._session_id = session_id or _make_session_id()
        self._device_id  = DEVICE_ID
        self._q: queue.Queue = queue.Queue(maxsize=self.QUEUE_MAX)
        self._running    = True
        self._handlers   = {}   # command_type → callback(payload)

        if not self._enabled:
            log.info("Cloud logging disabled (no Supabase credentials in .env)")
            return

        try:
            import requests
            self._requests = requests
        except ImportError:
            log.warning("pip install requests  — cloud logging disabled")
            self._enabled = False
            return

        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="cloud"
        )
        self._thread.start()
        log.info("Cloud logging active  session=%s  device=%s",
                 self._session_id, self._device_id)

    # ── Public API ────────────────────────────────────────────────────────

    def register_command_handler(self, cmd_type: str, callback):
        """Register a handler for a remote command type."""
        self._handlers[cmd_type] = callback

    def log_alert(self, message: str, priority: int,
                  zone_states: dict, closest_class: str,
                  closest_region: str, closest_proximity: str):
        if not self._enabled:
            return
        self._enqueue("alerts", {
            "session_id":        self._session_id,
            "device_id":         self._device_id,
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
        if not self._enabled:
            return
        self._enqueue("sensor_readings", {
            "session_id":      self._session_id,
            "device_id":       self._device_id,
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
        batch:     dict[str, list] = {}
        last_flush = time.time()
        last_cmd   = 0.0

        while self._running:
            try:
                item = self._q.get(timeout=0.5)
                t    = item["table"]
                batch.setdefault(t, []).append(item["row"])
            except queue.Empty:
                pass

            now = time.time()

            should_flush = (
                any(len(v) >= self.BATCH_SIZE for v in batch.values())
                or (now - last_flush >= self.FLUSH_INTERVAL and batch)
            )
            if should_flush:
                for table, rows in list(batch.items()):
                    if rows:
                        self._send(table, rows)
                        batch[table] = []
                batch      = {k: v for k, v in batch.items() if v}
                last_flush = now

            if now - last_cmd >= self.COMMAND_INTERVAL:
                self._poll_commands()
                last_cmd = now

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
                log.warning("Supabase %s: %d %s", table, r.status_code,
                            r.text[:80])
        except Exception as e:
            log.debug("Cloud send error: %s", e)

    def _poll_commands(self):
        """Check for pending remote commands and execute registered handlers."""
        try:
            url    = f"{SUPABASE_URL}/rest/v1/commands"
            params = (f"select=*&device_id=eq.{self._device_id}"
                      f"&executed=is.false&order=created_at.asc&limit=5")
            r = self._requests.get(
                f"{url}?{params}",
                headers={
                    "apikey":        SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                },
                timeout=5,
            )
            if r.status_code != 200:
                return

            for cmd in r.json():
                cmd_type = cmd.get("command", "")
                payload  = cmd.get("payload") or {}
                if isinstance(payload, str):
                    import json
                    try:
                        payload = json.loads(payload)
                    except Exception:
                        payload = {}

                log.info("Remote command: %s  payload=%s", cmd_type, payload)

                handler = self._handlers.get(cmd_type)
                if handler:
                    try:
                        handler(payload)
                    except Exception as e:
                        log.error("Command handler error: %s", e)
                else:
                    log.warning("No handler for command: %s", cmd_type)

                # Mark executed
                self._requests.patch(
                    f"{url}?id=eq.{cmd['id']}",
                    json={
                        "executed":    True,
                        "executed_at": _iso_now(),
                    },
                    headers={
                        "apikey":        SUPABASE_KEY,
                        "Authorization": f"Bearer {SUPABASE_KEY}",
                        "Content-Type":  "application/json",
                    },
                    timeout=5,
                )
        except Exception as e:
            log.debug("Command poll error: %s", e)


def _make_session_id() -> str:
    import uuid, datetime
    return (datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_" + str(uuid.uuid4())[:6])


def _iso_now() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"