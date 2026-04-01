# sensor_fusion.py
#
# Fuses the ultrasonic sensor reading with YOLO visual detections to produce
# a single, reliable proximity verdict for the guidance engine.
#
# Why fusion matters:
#   • YOLO proximity is inferred from bounding-box size — unreliable when an
#     object is very close and the box fills the frame or flickers.
#   • The ultrasonic sensor gives ground-truth distance but has no class info,
#     no left/right awareness, and false-fires on reflective surfaces.
#   • Together they cross-validate each other.
#
# Fusion rules (in priority order):
#   1. CRITICAL OVERRIDE  — sensor < DIST_CRITICAL AND center occupied
#                           → trust sensor completely, emergency stop
#   2. SENSOR FLOOR       — sensor says close/critical but YOLO says clear
#                           → something is there YOLO missed (low obstacle,
#                             bag, dark environment). Warn anyway.
#   3. VISUAL ANCHOR      — YOLO says occupied but sensor reads far
#                           → likely sensor noise / reflection. Trust camera,
#                             downgrade urgency slightly.
#   4. AGREEMENT          — both agree → normal confidence, use camera class/region
#   5. SENSOR SMOOTHING   — short-duration spikes (< SPIKE_WINDOW) are ignored
#                           to avoid false critical alerts from reflections.

import time
import logging
from dataclasses import dataclass
from ultrasonic import DIST_CRITICAL, DIST_CLOSE, DIST_MEDIUM

log = logging.getLogger("fusion")

# How many seconds a sensor reading must stay in a band before we trust it
# as a genuine reading rather than a spike
SPIKE_WINDOW = 0.4   # seconds


@dataclass
class FusedReading:
    # "critical" | "close" | "medium" | "far" | "none"
    proximity:       str
    # Where the source of truth is coming from
    source:          str   # "sensor" | "camera" | "both" | "sensor_floor"
    # Raw values for logging / HUD
    sensor_cm:       float | None
    visual_proximity: str
    # If True, the sensor is providing signal that the camera missed
    sensor_floor_active: bool = False
    # If True, the sensor spike filter suppressed a critical reading
    spike_suppressed: bool = False


class SensorFusion:
    def __init__(self):
        # Spike filter state
        self._band_start:    dict[str, float] = {}  # band → timestamp first seen
        self._current_band:  str = "none"

    def fuse(self, sensor_cm: float | None, scene_state) -> FusedReading:
        """
        Parameters
        ----------
        sensor_cm   : raw distance from UltrasonicSensor.read_distance_cm()
        scene_state : SceneState from SceneAnalyzer

        Returns a FusedReading with a single reliable proximity verdict.
        """
        now           = time.time()
        visual_prox   = scene_state.closest_proximity   # "close"|"medium"|"far"|"none"
        center_status = scene_state.zones["center"]
        center_occ    = center_status in ("occupied", "crowded")

        # Convert raw cm to band
        raw_sensor_band = self._cm_to_band(sensor_cm)

        # ── Spike filter ───────────────────────────────────────────────
        # Only trust a sensor band if it has been stable for SPIKE_WINDOW
        stable_band = self._stable_sensor_band(raw_sensor_band, now)

        # ── Rule 1: Critical override ──────────────────────────────────
        # Sensor is critical AND center is visually occupied — definite danger
        if stable_band == "critical" and center_occ:
            log.debug("FUSION critical override  sensor=%.0f cm", sensor_cm)
            return FusedReading(
                proximity        = "critical",
                source           = "both",
                sensor_cm        = sensor_cm,
                visual_proximity = visual_prox,
            )

        # ── Rule 2: Sensor floor ───────────────────────────────────────
        # Sensor detects close/critical but camera sees nothing — low obstacle
        # or lighting issue. Trust the sensor.
        if stable_band in ("critical", "close") and visual_prox in ("far", "none"):
            log.debug("FUSION sensor floor  sensor=%.0f cm  visual=%s",
                      sensor_cm, visual_prox)
            return FusedReading(
                proximity            = stable_band,
                source               = "sensor_floor",
                sensor_cm            = sensor_cm,
                visual_proximity     = visual_prox,
                sensor_floor_active  = True,
            )

        # ── Rule 3: Visual anchor (sensor noise) ──────────────────────
        # Camera sees a clear center but sensor fires — likely reflection/noise
        # Downgrade: use camera reading (far/none), but note the discrepancy
        if stable_band in ("critical", "close") and not center_occ:
            log.debug("FUSION visual anchor (sensor noise likely)  "
                      "sensor=%.0f cm  center=%s", sensor_cm, center_status)
            # Still use camera proximity for side objects
            return FusedReading(
                proximity        = visual_prox,
                source           = "camera",
                sensor_cm        = sensor_cm,
                visual_proximity = visual_prox,
            )

        # ── Rule 4: Spike suppressed ───────────────────────────────────
        if raw_sensor_band in ("critical", "close") and stable_band != raw_sensor_band:
            log.debug("FUSION spike suppressed  raw_band=%s", raw_sensor_band)
            return FusedReading(
                proximity         = visual_prox,
                source            = "camera",
                sensor_cm         = sensor_cm,
                visual_proximity  = visual_prox,
                spike_suppressed  = True,
            )

        # ── Rule 5: Agreement / default ───────────────────────────────
        # Use the more alarming of camera and sensor (both are credible here)
        cam_rank    = _PROX_RANK.get(visual_prox, 0)
        sensor_rank = _PROX_RANK.get(stable_band, 0)

        if sensor_rank >= cam_rank:
            final = stable_band
            src   = "sensor" if sensor_rank > cam_rank else "both"
        else:
            final = visual_prox
            src   = "camera"

        return FusedReading(
            proximity        = final,
            source           = src,
            sensor_cm        = sensor_cm,
            visual_proximity = visual_prox,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _cm_to_band(self, cm: float | None) -> str:
        if cm is None:
            return "none"
        if cm < DIST_CRITICAL:
            return "critical"
        if cm < DIST_CLOSE:
            return "close"
        if cm < DIST_MEDIUM:
            return "medium"
        return "far"

    def _stable_sensor_band(self, band: str, now: float) -> str:
        """
        Returns the band only once it has been continuously reported for
        SPIKE_WINDOW seconds.  New bands start a timer; if they persist
        they become trusted.
        """
        if band != self._current_band:
            # Band changed — start timer for new band
            if band not in self._band_start:
                self._band_start[band] = now
            # Check if the new band has been seen long enough
            if now - self._band_start.get(band, now) >= SPIKE_WINDOW:
                self._current_band = band
                self._band_start   = {band: self._band_start[band]}
        else:
            # Same band — update start time bucket, it's stable
            self._band_start = {band: self._band_start.get(band, now)}

        return self._current_band


_PROX_RANK = {"none": 0, "far": 1, "medium": 2, "close": 3, "critical": 4}