# shared_state.py
#
# Thread-safe bridge between the navigation loop (main.py) and the
# web server (server.py). Both import this module and share the same
# singleton instance because Python modules are only loaded once per process.
#
# main.py  → calls update_frame() and update_state() every loop iteration
# server.py → reads latest_jpeg() and get_state() to serve the browser

import threading
import time
import cv2

_lock        = threading.Lock()
_jpeg_bytes  = b""
_state       = {
    "zones":            {"left": "clear", "center": "clear", "right": "clear"},
    "zone_counts":      {"left": 0, "center": 0, "right": 0},
    "closest_proximity": "none",
    "closest_class":    "",
    "closest_region":   "",
    "sensor_cm":        None,
    "sensor_band":      "none",
    "sensor_source":    "none",
    "sensor_floor":     False,
    "last_message":     "",
    "last_priority":    0,
    "alert_count":      0,
    "object_count":     0,
    "confirmed_count":  0,
    "uptime_start":     time.time(),
}


def update_frame(frame):
    """Call from main loop with the annotated BGR frame (numpy array)."""
    global _jpeg_bytes
    ok, buf = cv2.imencode(
        ".jpg", frame,
        [cv2.IMWRITE_JPEG_QUALITY, 70]   # 70% quality — good balance for streaming
    )
    if ok:
        with _lock:
            _jpeg_bytes = buf.tobytes()


def update_state(scene_state, detections, message, priority, fused=None):
    """Call from main loop after guidance.decide() returns."""
    global _state
    confirmed = sum(1 for d in detections if d["confirmed"])

    with _lock:
        _state["zones"]             = dict(scene_state.zones)
        _state["zone_counts"]       = dict(scene_state.zone_counts)
        _state["closest_proximity"] = scene_state.closest_proximity
        _state["closest_class"]     = scene_state.closest_class
        _state["closest_region"]    = scene_state.closest_region
        _state["object_count"]      = len(detections)
        _state["confirmed_count"]   = confirmed

        if fused is not None:
            _state["sensor_cm"]     = round(fused.sensor_cm, 1) if fused.sensor_cm else None
            _state["sensor_band"]   = fused.proximity
            _state["sensor_source"] = fused.source
            _state["sensor_floor"]  = fused.sensor_floor_active
        else:
            _state["sensor_cm"]     = None
            _state["sensor_band"]   = "none"
            _state["sensor_source"] = "none"
            _state["sensor_floor"]  = False

        if message:
            _state["last_message"]  = message
            _state["last_priority"] = priority
            _state["alert_count"]  += 1


def latest_jpeg() -> bytes:
    with _lock:
        return _jpeg_bytes


def get_state() -> dict:
    with _lock:
        s = dict(_state)
    s["uptime_seconds"] = int(time.time() - s["uptime_start"])
    return s
