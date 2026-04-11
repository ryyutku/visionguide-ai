# shared_state.py
#
# Thread-safe bridge between main.py (writes) and server.py (reads).
# Initialised with a placeholder JPEG so the video stream never
# blocks waiting for the first real frame.

import threading
import time
import cv2
import numpy as np

_lock       = threading.Lock()
_state_lock = threading.Lock()

# ── Generate a placeholder frame so /video works immediately on startup ──────
def _make_placeholder() -> bytes:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Waiting for camera...",
                (140, 240), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (80, 80, 80), 2, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


_jpeg_bytes: bytes = _make_placeholder()

_state = {
    "zones":             {"left": "clear", "center": "clear", "right": "clear"},
    "zone_counts":       {"left": 0, "center": 0, "right": 0},
    "closest_proximity": "none",
    "closest_class":     "",
    "closest_region":    "",
    "sensor_cm":         None,
    "sensor_band":       "none",
    "sensor_source":     "none",
    "sensor_floor":      False,
    "last_message":      "",
    "last_priority":     0,
    "alert_count":       0,
    "object_count":      0,
    "confirmed_count":   0,
    "uptime_start":      time.time(),
}


def update_frame(frame):
    global _jpeg_bytes
    ok, buf = cv2.imencode(
        ".jpg", frame,
        [cv2.IMWRITE_JPEG_QUALITY, 75]
    )
    if ok:
        with _lock:
            _jpeg_bytes = buf.tobytes()


def update_state(scene_state, detections, message, priority, fused=None):
    confirmed = sum(1 for d in detections if d["confirmed"])
    with _state_lock:
        _state["zones"]             = dict(scene_state.zones)
        _state["zone_counts"]       = dict(scene_state.zone_counts)
        _state["closest_proximity"] = scene_state.closest_proximity
        _state["closest_class"]     = scene_state.closest_class
        _state["closest_region"]    = scene_state.closest_region
        _state["object_count"]      = len(detections)
        _state["confirmed_count"]   = confirmed

        if fused is not None:
            _state["sensor_cm"]     = (round(fused.sensor_cm, 1)
                                       if fused.sensor_cm is not None else None)
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
    with _state_lock:
        s = dict(_state)
        s["zones"]       = dict(s["zones"])
        s["zone_counts"] = dict(s["zone_counts"])
    s["uptime_seconds"] = int(time.time() - s["uptime_start"])
    return s