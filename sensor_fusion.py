# sensor_fusion.py
#
# Fuses ultrasonic sensor distance with YOLO visual detections.
#
# The sensor and camera have complementary blind spots:
#   Camera blind spots : very close objects (bbox fills frame → flicker),
#                        low objects (bags, steps), darkness
#   Sensor blind spots : no class info, no left/right awareness,
#                        false fires on glass/reflective surfaces
#
# Fusion rules (applied in order):
#
#   Rule 1 — CRITICAL + CAMERA OCCUPIED
#     Sensor < 40cm AND camera sees something in any zone
#     → Both agree something is very close. Emergency stop.
#
#   Rule 2 — SENSOR FLOOR
#     Sensor < 80cm BUT camera sees nothing anywhere
#     → Undetected obstacle (low object, bad lighting). Warn on sensor alone.
#
#   Rule 3 — VISUAL ANCHOR  (sensor noise suppression)
#     Sensor fires close/critical BUT camera sees all zones as clear
#     → Likely glass door, reflective floor, sensor noise.
#       Trust camera (ignore sensor).
#
#   Rule 4 — SPIKE FILTER
#     Sensor band changed but hasn't been stable for SPIKE_WINDOW seconds yet
#     → Too short to trust. Keep previous stable band.
#
#   Rule 5 — AGREEMENT / DEFAULT
#     Neither an override nor a conflict → take the more alarming of the two.

import time
import logging
from dataclasses import dataclass, field
from ultrasonic import DIST_CRITICAL, DIST_CLOSE, DIST_MEDIUM

log = logging.getLogger("fusion")

SPIKE_WINDOW = 0.4   # seconds a reading must be stable before trusted

_PROX_RANK = {"none": 0, "far": 1, "medium": 2, "close": 3, "critical": 4}


@dataclass
class FusedReading:
    proximity:           str          # "critical"|"close"|"medium"|"far"|"none"
    source:              str          # "sensor"|"camera"|"both"|"sensor_floor"
    sensor_cm:           float | None
    visual_proximity:    str
    sensor_floor_active: bool = False
    spike_suppressed:    bool = False


class SensorFusion:
    def __init__(self):
        self._band_start:   dict[str, float] = {}
        self._current_band: str = "none"

    def fuse(self, sensor_cm: float | None, scene_state) -> FusedReading:
        now         = time.time()
        visual_prox = scene_state.closest_proximity  # closest object in any zone

        # Is ANYTHING detected by the camera, in any zone?
        any_occupied = any(
            v in ("occupied", "crowded")
            for v in scene_state.zones.values()
        )

        raw_band    = self._cm_to_band(sensor_cm)
        stable_band = self._stable_band(raw_band, now)

        # ── Rule 1: Critical override ──────────────────────────────────────
        # Sensor very close AND camera confirms something is there
        if stable_band == "critical" and any_occupied:
            log.debug("FUSION rule1 critical+camera  sensor=%.0fcm", sensor_cm or 0)
            return FusedReading(
                proximity="critical", source="both",
                sensor_cm=sensor_cm, visual_proximity=visual_prox,
            )

        # ── Rule 2: Sensor floor ───────────────────────────────────────────
        # Sensor close/critical but camera sees absolutely nothing
        # Something real is there that YOLO missed
        if stable_band in ("critical", "close") and not any_occupied:
            log.debug("FUSION rule2 sensor floor  sensor=%.0fcm", sensor_cm or 0)
            return FusedReading(
                proximity=stable_band, source="sensor_floor",
                sensor_cm=sensor_cm, visual_proximity=visual_prox,
                sensor_floor_active=True,
            )

        # ── Rule 3: Visual anchor — sensor noise ──────────────────────────
        # Sensor fires close/critical but ALL camera zones are clear
        # (nothing visible anywhere in the frame — likely reflection/glass)
        # Fixed vs old version: we check any_occupied not just center_occ,
        # so a real side object won't be discarded as noise.
        if stable_band in ("critical", "close") and not any_occupied:
            # This branch is now unreachable (Rule 2 catches it first) but
            # kept as documentation. If sensor fires and camera sees nothing
            # at all → sensor floor (Rule 2) already handles it.
            pass

        # Sensor fires but camera sees SOMETHING (just not agreeing on proximity)
        # e.g. sensor says close but camera says far for the same object
        # → trust the sensor (it's more reliable for distance)
        if stable_band in ("critical", "close") and any_occupied:
            # Already handled by Rule 1 for critical. For "close":
            if stable_band == "close":
                log.debug("FUSION rule3b sensor+camera agree close  sensor=%.0fcm",
                          sensor_cm or 0)
                return FusedReading(
                    proximity="close", source="both",
                    sensor_cm=sensor_cm, visual_proximity=visual_prox,
                )

        # ── Rule 4: Spike filter ───────────────────────────────────────────
        if raw_band in ("critical", "close") and stable_band != raw_band:
            log.debug("FUSION rule4 spike suppressed  raw=%s stable=%s",
                      raw_band, stable_band)
            return FusedReading(
                proximity=visual_prox, source="camera",
                sensor_cm=sensor_cm, visual_proximity=visual_prox,
                spike_suppressed=True,
            )

        # ── Rule 5: Agreement / default ────────────────────────────────────
        cam_rank    = _PROX_RANK.get(visual_prox, 0)
        sensor_rank = _PROX_RANK.get(stable_band, 0)

        if sensor_rank >= cam_rank:
            return FusedReading(
                proximity=stable_band,
                source="sensor" if sensor_rank > cam_rank else "both",
                sensor_cm=sensor_cm, visual_proximity=visual_prox,
            )
        return FusedReading(
            proximity=visual_prox, source="camera",
            sensor_cm=sensor_cm, visual_proximity=visual_prox,
        )

    def _cm_to_band(self, cm: float | None) -> str:
        if cm is None:
            return "none"
        if cm < DIST_CRITICAL: return "critical"
        if cm < DIST_CLOSE:    return "close"
        if cm < DIST_MEDIUM:   return "medium"
        return "far"

    def _stable_band(self, band: str, now: float) -> str:
        """Only trust a band once it has been stable for SPIKE_WINDOW seconds."""
        if band != self._current_band:
            if band not in self._band_start:
                self._band_start[band] = now
            if now - self._band_start.get(band, now) >= SPIKE_WINDOW:
                self._current_band = band
                self._band_start   = {band: self._band_start[band]}
        else:
            self._band_start = {band: self._band_start.get(band, now)}
        return self._current_band
