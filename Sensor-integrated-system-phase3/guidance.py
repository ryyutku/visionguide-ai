# guidance.py  —  Navigation guidance with sensor fusion support
#
# Accepts an optional FusedReading from sensor_fusion.py.
# When fusion data is present:
#   • Critical sensor distance immediately overrides everything → "Stop!"
#   • Sensor floor (unseen obstacle) triggers a generic "obstacle ahead" warn
#   • Sensor distance used to decide whether a "path clear" is trustworthy
#     (only clear if BOTH camera center is empty AND sensor reads > DIST_CLOSE)
#   • Proximity escalation uses real cm distance, not just bbox size band
#
# Without fusion data (sensor=None) the engine falls back to camera-only
# behaviour — identical to the laptop version.

import time
import logging
from dataclasses import dataclass, field
from scene import SceneState

log = logging.getLogger("guidance")

PRIORITY_HIGH   = 3
PRIORITY_MEDIUM = 2
PRIORITY_LOW    = 1

_PROX_RANK = {"far": 1, "medium": 2, "close": 3, "critical": 4, "none": 0}


@dataclass
class _ObjectMemory:
    track_id:           int
    obj_class:          str
    region:             str
    last_proximity:     str   = "far"
    announced_at:       float = 0.0
    announcement_count: int   = 0
    first_seen:         float = field(default_factory=time.time)
    last_seen:          float = field(default_factory=time.time)


class GuidanceEngine:

    # ── Timing constants ──────────────────────────────────────────────────
    COOLDOWN_URGENT  = 6.0
    COOLDOWN_WARNING = 7.0
    COOLDOWN_CLEAR   = 10.0

    # Center must stay continuously empty for this long before saying "clear"
    CLEAR_SUSTAIN_SECONDS = 2.5

    # When sensor is present, require it to also read > DIST_CLOSE before clear
    # (prevents false clears when sensor sees something camera missed)
    CLEAR_SENSOR_MIN_CM = 90    # cm — must be beyond DIST_CLOSE (80 cm)

    MIN_REANNOUNCE_GAP  = 5.0
    OBJECT_GRACE_PERIOD = 2.5

    QUEUE_MODE_THRESHOLD    = 8.0
    QUEUE_REMINDER_INTERVAL = 20.0

    # How often to repeat the "obstacle ahead" sensor-floor warning
    SENSOR_FLOOR_COOLDOWN = 5.0

    def __init__(self):
        self._objects: dict[int, _ObjectMemory] = {}

        self._last_urgent:       float = 0.0
        self._last_warning:      float = 0.0
        self._last_clear:        float = 0.0
        self._last_sensor_floor: float = 0.0

        self._was_blocked:         bool  = False
        self._block_start:         float = 0.0
        self._cleared_said:        bool  = False
        self._prev_center_ids:     set   = set()
        self._center_empty_since:  float | None = None

    # ── Public ────────────────────────────────────────────────────────────

    def decide(self,
               state:      SceneState,
               detections: list,
               speech=None,
               fused=None   # FusedReading | None
               ) -> tuple[str | None, int]:
        """
        fused : FusedReading from SensorFusion.fuse(), or None for camera-only.
        """
        now = time.time()
        self._sync_memory(detections, now)

        # ── 0. CRITICAL sensor override ──────────────────────────────────
        if fused is not None and fused.proximity == "critical":
            cm  = fused.sensor_cm
            msg = f"Stop! Obstacle {int(cm)} centimetres ahead" if cm else "Stop! Obstacle ahead"
            if now - self._last_urgent >= 3.0:   # don't scream every frame
                self._last_urgent = now
                log.warning("CRITICAL  %s  [sensor %.0f cm]", msg, cm or 0)
                if speech:
                    speech.say_urgent(msg)
                return msg, PRIORITY_HIGH
            return None, 0

        # ── 1. SENSOR FLOOR — unseen obstacle ────────────────────────────
        if fused is not None and fused.sensor_floor_active:
            if now - self._last_sensor_floor >= self.SENSOR_FLOOR_COOLDOWN:
                cm  = fused.sensor_cm
                msg = (f"Obstacle ahead, {int(cm)} centimetres"
                       if cm else "Obstacle ahead, not visible")
                self._last_sensor_floor = now
                log.info("SENSOR FLOOR  %s", msg)
                if speech:
                    speech.say_urgent(msg)
                return msg, PRIORITY_HIGH
            return None, 0

        center_blocked = state.zones["center"] in ("occupied", "crowded")

        # ── 2. CENTER BLOCKED ────────────────────────────────────────────
        if center_blocked:
            self._center_empty_since = None
            self._cleared_said       = False
            msg, pri = self._handle_center_blocked(state, detections, now,
                                                   speech, fused)
            if msg:
                return msg, pri
            return None, 0

        # ── 3. CENTER EMPTY — sustain check ─────────────────────────────
        if self._center_empty_since is None:
            self._center_empty_since = now
        time_empty = now - self._center_empty_since

        if self._was_blocked and time_empty >= self.CLEAR_SUSTAIN_SECONDS:
            # Also require sensor to agree (if available)
            sensor_ok = (
                fused is None
                or fused.sensor_cm is None
                or fused.sensor_cm >= self.CLEAR_SENSOR_MIN_CM
            )
            if sensor_ok and not self._cleared_said:
                self._was_blocked     = False
                self._block_start     = 0.0
                self._prev_center_ids = set()
                self._cleared_said    = True
                self._last_clear      = now
                log.info("CLEAR  Path clear, move forward  [sustained %.1fs]",
                         time_empty)
                return "Path clear, move forward", PRIORITY_LOW

        if not center_blocked:
            self._was_blocked = False

        # ── 4. SIDE WARNING ──────────────────────────────────────────────
        msg, pri = self._handle_side_warning(state, detections, now, fused)
        if msg:
            return msg, pri

        # ── 5. PERIODIC CLEAR ────────────────────────────────────────────
        sensor_ok = (
            fused is None
            or fused.sensor_cm is None
            or fused.sensor_cm >= self.CLEAR_SENSOR_MIN_CM
        )
        if (now - self._last_clear >= self.COOLDOWN_CLEAR
                and self._was_previously_busy(now)
                and time_empty >= self.CLEAR_SUSTAIN_SECONDS
                and sensor_ok):
            self._last_clear = now
            return "Path clear", PRIORITY_LOW

        return None, 0

    # ── Internal ──────────────────────────────────────────────────────────

    def _handle_center_blocked(self, state, detections, now, speech, fused):
        center_ids = {
            d["id"] for d in detections
            if d["confirmed"] and d["region"] == "center"
        }

        new_ids = set()
        for tid in center_ids - self._prev_center_ids:
            mem = self._objects.get(tid)
            if mem is None:
                continue
            if now - mem.announced_at >= self.MIN_REANNOUNCE_GAP:
                new_ids.add(tid)

        if not self._was_blocked:
            self._was_blocked  = True
            self._block_start  = now
            self._cleared_said = False

        self._prev_center_ids = center_ids
        time_blocked = now - self._block_start

        # New object entered center
        if new_ids:
            msg = self._route_around(state)
            self._last_urgent = now
            for tid in new_ids:
                if tid in self._objects:
                    self._objects[tid].announced_at        = now
                    self._objects[tid].announcement_count += 1
            log.info("URGENT  %s  [new center id=%s]", msg, new_ids)
            if speech:
                speech.say_urgent(msg)
            return msg, PRIORITY_HIGH

        # Proximity escalation — prefer real cm distance when available
        eid, emem = self._check_proximity_escalation(
            center_ids, detections, fused
        )
        if eid is not None:
            cm_str = (f", {int(fused.sensor_cm)} centimetres away"
                      if fused and fused.sensor_cm else "")
            msg = f"{emem.obj_class} getting closer ahead{cm_str}, stop"
            self._last_urgent          = now
            emem.announced_at          = now
            emem.announcement_count   += 1
            log.info("URGENT  %s  [escalation id=%d]", msg, eid)
            if speech:
                speech.say_urgent(msg)
            return msg, PRIORITY_HIGH

        # Queue mode
        if time_blocked >= self.QUEUE_MODE_THRESHOLD:
            if now - self._last_urgent >= self.QUEUE_REMINDER_INTERVAL:
                obj  = state.closest_class or "obstacle"
                msg  = f"{obj} still ahead, wait or find alternate route"
                self._last_urgent = now
                log.info("URGENT  %s  [queue mode]", msg)
                if speech:
                    speech.say_urgent(msg)
                return msg, PRIORITY_HIGH
            return None, 0

        # Normal repeat
        if now - self._last_urgent >= self.COOLDOWN_URGENT:
            msg = self._route_around(state)
            self._last_urgent = now
            log.info("URGENT  %s  [repeat]", msg)
            if speech:
                speech.say_urgent(msg)
            return msg, PRIORITY_HIGH

        return None, 0

    def _handle_side_warning(self, state, detections, now, fused):
        if state.closest_proximity != "close":
            return None, 0
        if state.closest_region not in ("left", "right"):
            return None, 0

        for d in detections:
            if (d["confirmed"]
                    and d["proximity"] == "close"
                    and d["region"] == state.closest_region):

                tid = d["id"]
                mem = self._objects.get(tid)
                if mem is None:
                    continue

                prev_rank      = _PROX_RANK.get(mem.last_proximity, 0)
                curr_rank      = _PROX_RANK["close"]
                just_escalated = curr_rank > prev_rank
                first_alert    = mem.announcement_count == 0
                timed_repeat   = now - mem.announced_at >= self.COOLDOWN_WARNING
                global_ok      = now - self._last_warning >= self.COOLDOWN_WARNING

                if (just_escalated or first_alert) and global_ok:
                    side  = state.closest_region
                    other = "right" if side == "left" else "left"
                    msg   = f"{d['class']} close on your {side}, move {other}"
                    self._last_warning      = now
                    mem.last_proximity      = "close"
                    mem.announced_at        = now
                    mem.announcement_count += 1
                    log.info("WARNING  %s  [id=%d]", msg, tid)
                    return msg, PRIORITY_MEDIUM

                elif timed_repeat and global_ok:
                    side  = state.closest_region
                    other = "right" if side == "left" else "left"
                    msg   = f"{d['class']} on your {side}"
                    self._last_warning = now
                    mem.announced_at   = now
                    log.info("WARNING  %s  [timed repeat id=%d]", msg, tid)
                    return msg, PRIORITY_MEDIUM

        return None, 0

    def _check_proximity_escalation(self, center_ids, detections, fused):
        for d in detections:
            if d["id"] not in center_ids or not d["confirmed"]:
                continue
            tid = d["id"]
            mem = self._objects.get(tid)
            if mem is None:
                continue

            # Use real sensor distance band when available
            if fused and fused.sensor_cm is not None:
                from ultrasonic import DIST_CRITICAL, DIST_CLOSE, DIST_MEDIUM
                cm = fused.sensor_cm
                if   cm < DIST_CRITICAL: curr_band = "critical"
                elif cm < DIST_CLOSE:    curr_band = "close"
                elif cm < DIST_MEDIUM:   curr_band = "medium"
                else:                    curr_band = "far"
            else:
                curr_band = d["proximity"]

            prev_rank = _PROX_RANK.get(mem.last_proximity, 0)
            curr_rank = _PROX_RANK.get(curr_band, 0)
            if curr_rank > prev_rank:
                mem.last_proximity = curr_band
                return tid, mem

        return None, None

    def _sync_memory(self, detections, now):
        active_ids = set()
        for d in detections:
            if not d["confirmed"]:
                continue
            tid = d["id"]
            active_ids.add(tid)
            if tid not in self._objects:
                self._objects[tid] = _ObjectMemory(
                    track_id       = tid,
                    obj_class      = d["class"],
                    region         = d["region"],
                    last_proximity = d["proximity"],
                    first_seen     = now,
                    last_seen      = now,
                )
            else:
                self._objects[tid].region    = d["region"]
                self._objects[tid].last_seen = now

        to_delete = [
            tid for tid, mem in self._objects.items()
            if tid not in active_ids
            and now - mem.last_seen > self.OBJECT_GRACE_PERIOD
        ]
        for tid in to_delete:
            del self._objects[tid]
        self._prev_center_ids -= set(to_delete)

    def _was_previously_busy(self, now) -> bool:
        return (now - self._last_urgent  < 30.0 or
                now - self._last_warning < 30.0)

    def _route_around(self, state: SceneState) -> str:
        ls  = self._zone_score(state.zones["left"])
        rs  = self._zone_score(state.zones["right"])
        obj = state.closest_class or "obstacle"
        if ls == 0 and rs == 0:
            d = "left" if state.zone_counts["left"] <= state.zone_counts["right"] else "right"
            return f"{obj} ahead, move {d}"
        if ls < rs:
            return f"{obj} ahead, move left"
        if rs < ls:
            return f"{obj} ahead, move right"
        return f"{obj} ahead, stop and wait"

    @staticmethod
    def _zone_score(status: str) -> int:
        return {"clear": 0, "occupied": 1, "crowded": 2}.get(status, 1)