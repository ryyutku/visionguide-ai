# guidance.py
#
# Alert hierarchy — strictly one alert at a time, highest wins:
#
#   P3 STOP      — sensor < STOP_DISTANCE_CM (75cm)
#                  Matches DIST_CRITICAL in ultrasonic.py.
#                  Bypasses everything including queue mode.
#                  Cooldown: 3s.
#
#   P3 URGENT    — new confirmed object entered center zone
#                  OR existing center object escalated closer
#                  Cooldown: 6s for repeats
#
#   P2 WARNING   — close object on left/right side
#                  Only fires once per object per proximity band
#                  Cooldown: 7s
#
#   P1 CLEAR     — center empty for CLEAR_SUSTAIN seconds
#                  AND sensor reads > CLEAR_SENSOR_CM (120cm)
#                  Only fires if something was previously blocking
#
# Threshold rationale:
#   Pipeline latency ~250-350ms on Pi 4. At walking pace 1.2 m/s
#   the user moves ~36cm during processing. Thresholds are set
#   ~35cm higher than desired real-world warning distances so
#   alerts arrive on time.
#   STOP_DISTANCE_CM = 75  (alert arrives when object ~40cm away)
#   CLEAR_SENSOR_CM  = 120 (must be > DIST_CLOSE=110 to avoid false clears)

import time
import logging
from dataclasses import dataclass, field
from scene import SceneState

log = logging.getLogger("guidance")

PRIORITY_HIGH   = 3
PRIORITY_MEDIUM = 2
PRIORITY_LOW    = 1

_PROX_RANK = {"none": 0, "far": 1, "medium": 2, "close": 3, "critical": 4}

# ── Key thresholds — must stay consistent with ultrasonic.py ─────────────────
# DIST_CRITICAL = 75cm in ultrasonic.py  →  STOP_DISTANCE_CM must equal it
# DIST_CLOSE    = 110cm in ultrasonic.py →  CLEAR_SENSOR_CM must be > 110
STOP_DISTANCE_CM = 75    # cm — sensor hard stop (latency-compensated)
CLEAR_SENSOR_CM  = 120   # cm — sensor must read beyond this before "path clear"


@dataclass
class _Obj:
    track_id:           int
    obj_class:          str
    region:             str
    last_proximity:     str   = "far"
    announced_at:       float = 0.0
    announcement_count: int   = 0
    first_seen:         float = field(default_factory=time.time)
    last_seen:          float = field(default_factory=time.time)


class GuidanceEngine:

    STOP_COOLDOWN         = 3.0
    COOLDOWN_URGENT       = 6.0
    COOLDOWN_WARNING      = 7.0
    COOLDOWN_CLEAR        = 12.0
    SENSOR_FLOOR_COOLDOWN = 6.0

    # While a HIGH alert fired within this window, suppress lower alerts
    ACTIVE_WINDOW = 4.0

    # Center must be empty this long before "path clear" fires
    CLEAR_SUSTAIN  = 2.5

    # Same object stays silent for this long after announcement
    MIN_REANNOUNCE = 5.0

    # Object kept in memory this long after disappearing (YOLO flicker)
    GRACE_PERIOD   = 2.5

    # Queue mode — object stationary in center
    QUEUE_THRESHOLD = 8.0
    QUEUE_INTERVAL  = 20.0

    def __init__(self):
        self._objects: dict[int, _Obj] = {}

        self._last_stop:    float = 0.0
        self._last_urgent:  float = 0.0
        self._last_warning: float = 0.0
        self._last_clear:   float = 0.0
        self._last_floor:   float = 0.0

        self._was_blocked:        bool        = False
        self._block_start:        float       = 0.0
        self._cleared_said:       bool        = False
        self._prev_center_ids:    set         = set()
        self._center_empty_since: float | None = None

    # ── Public ────────────────────────────────────────────────────────────

    def decide(self, state, detections, speech=None, fused=None):
        now = time.time()
        self._sync(detections, now)

        # ── LEVEL 0: Hard stop (sensor < 75cm) ───────────────────────────
        if fused is not None and fused.sensor_cm is not None:
            if fused.sensor_cm < STOP_DISTANCE_CM:
                if now - self._last_stop >= self.STOP_COOLDOWN:
                    self._last_stop   = now
                    self._last_urgent = now
                    cm  = int(fused.sensor_cm)
                    msg = f"Stop, obstacle {cm} centimetres ahead"
                    log.warning("STOP  %s", msg)
                    if speech:
                        speech.say_urgent(msg)
                    return msg, PRIORITY_HIGH
                return None, 0

        last_high = max(self._last_stop, self._last_urgent)

        # ── LEVEL 1: Sensor floor (unseen obstacle) ───────────────────────
        if fused is not None and fused.sensor_floor_active:
            if now - self._last_floor >= self.SENSOR_FLOOR_COOLDOWN:
                self._last_floor  = now
                self._last_urgent = now
                cm  = fused.sensor_cm
                msg = (f"Obstacle ahead, {int(cm)} centimetres"
                       if cm else "Obstacle ahead")
                log.info("FLOOR  %s", msg)
                if speech:
                    speech.say_urgent(msg)
                return msg, PRIORITY_HIGH
            return None, 0

        center_blocked = state.zones["center"] in ("occupied", "crowded")

        # ── LEVEL 2: Center blocked ───────────────────────────────────────
        if center_blocked:
            self._center_empty_since = None
            self._cleared_said       = False
            msg, pri = self._center_blocked(state, detections, now, speech, fused)
            if msg:
                return msg, pri
            return None, 0

        # ── Center empty — sustain check ─────────────────────────────────
        if self._center_empty_since is None:
            self._center_empty_since = now
        time_empty = now - self._center_empty_since

        if self._was_blocked and time_empty >= self.CLEAR_SUSTAIN:
            if self._sensor_ok(fused) and not self._cleared_said:
                self._was_blocked     = False
                self._block_start     = 0.0
                self._prev_center_ids = set()
                self._cleared_said    = True
                self._last_clear      = now
                log.info("CLEAR  Path clear")
                return "Path clear, move forward", PRIORITY_LOW

        if not center_blocked:
            self._was_blocked = False

        # ── LEVEL 3: Side warning (suppressed if HIGH fired recently) ─────
        if now - last_high > self.ACTIVE_WINDOW:
            msg, pri = self._side_warning(state, detections, now)
            if msg:
                return msg, pri

        # ── LEVEL 4: Periodic clear ───────────────────────────────────────
        if (now - last_high > self.ACTIVE_WINDOW
                and now - self._last_clear >= self.COOLDOWN_CLEAR
                and self._was_busy(now)
                and time_empty >= self.CLEAR_SUSTAIN
                and self._sensor_ok(fused)):
            self._last_clear = now
            return "Path clear", PRIORITY_LOW

        return None, 0

    # ── Internal ──────────────────────────────────────────────────────────

    def _center_blocked(self, state, detections, now, speech, fused):
        center_ids = {
            d["id"] for d in detections
            if d["confirmed"] and d["region"] == "center"
        }
        new_ids = {
            tid for tid in center_ids - self._prev_center_ids
            if tid in self._objects
            and now - self._objects[tid].announced_at >= self.MIN_REANNOUNCE
        }

        if not self._was_blocked:
            self._was_blocked  = True
            self._block_start  = now
            self._cleared_said = False

        self._prev_center_ids = center_ids
        time_blocked = now - self._block_start

        if new_ids:
            msg = self._route(state)
            self._last_urgent = now
            for tid in new_ids:
                if tid in self._objects:
                    self._objects[tid].announced_at        = now
                    self._objects[tid].announcement_count += 1
            log.info("URGENT  %s  [new %s]", msg, new_ids)
            if speech:
                speech.say_urgent(msg)
            return msg, PRIORITY_HIGH

        eid, emem = self._escalation(center_ids, detections, fused)
        if eid is not None:
            msg = f"{emem.obj_class} getting closer, stop"
            self._last_urgent        = now
            emem.announced_at        = now
            emem.announcement_count += 1
            log.info("URGENT  %s  [escalation id=%d]", msg, eid)
            if speech:
                speech.say_urgent(msg)
            return msg, PRIORITY_HIGH

        if time_blocked >= self.QUEUE_THRESHOLD:
            if now - self._last_urgent >= self.QUEUE_INTERVAL:
                obj = state.closest_class or "obstacle"
                msg = f"{obj} still ahead, wait"
                self._last_urgent = now
                log.info("URGENT  %s  [queue]", msg)
                if speech:
                    speech.say_urgent(msg)
                return msg, PRIORITY_HIGH
            return None, 0

        if now - self._last_urgent >= self.COOLDOWN_URGENT:
            msg = self._route(state)
            self._last_urgent = now
            log.info("URGENT  %s  [repeat]", msg)
            if speech:
                speech.say_urgent(msg)
            return msg, PRIORITY_HIGH

        return None, 0

    def _side_warning(self, state, detections, now):
        if state.closest_proximity != "close":
            return None, 0
        if state.closest_region not in ("left", "right"):
            return None, 0

        for d in detections:
            if (not d["confirmed"]
                    or d["proximity"] != "close"
                    or d["region"] != state.closest_region):
                continue
            tid = d["id"]
            mem = self._objects.get(tid)
            if mem is None:
                continue

            escalated = _PROX_RANK["close"] > _PROX_RANK.get(mem.last_proximity, 0)
            first     = mem.announcement_count == 0
            repeat_ok = now - mem.announced_at >= self.COOLDOWN_WARNING
            global_ok = now - self._last_warning >= self.COOLDOWN_WARNING
            side      = state.closest_region
            other     = "right" if side == "left" else "left"

            if (escalated or first) and global_ok:
                msg = f"{d['class']} close on your {side}, move {other}"
                self._last_warning      = now
                mem.last_proximity      = "close"
                mem.announced_at        = now
                mem.announcement_count += 1
                log.info("WARNING  %s", msg)
                return msg, PRIORITY_MEDIUM
            elif repeat_ok and global_ok:
                msg = f"{d['class']} on your {side}"
                self._last_warning = now
                mem.announced_at   = now
                log.info("WARNING  %s  [repeat]", msg)
                return msg, PRIORITY_MEDIUM

        return None, 0

    def _escalation(self, center_ids, detections, fused):
        for d in detections:
            if d["id"] not in center_ids or not d["confirmed"]:
                continue
            mem = self._objects.get(d["id"])
            if mem is None:
                continue
            if fused and fused.sensor_cm is not None:
                from ultrasonic import DIST_CRITICAL, DIST_CLOSE, DIST_MEDIUM
                cm = fused.sensor_cm
                band = ("critical" if cm < DIST_CRITICAL else
                        "close"    if cm < DIST_CLOSE    else
                        "medium"   if cm < DIST_MEDIUM   else "far")
            else:
                band = d["proximity"]
            if _PROX_RANK.get(band, 0) > _PROX_RANK.get(mem.last_proximity, 0):
                mem.last_proximity = band
                return d["id"], mem
        return None, None

    def _sync(self, detections, now):
        active = set()
        for d in detections:
            if not d["confirmed"]:
                continue
            tid = d["id"]
            active.add(tid)
            if tid not in self._objects:
                self._objects[tid] = _Obj(
                    track_id=tid, obj_class=d["class"],
                    region=d["region"], last_proximity=d["proximity"],
                    first_seen=now, last_seen=now,
                )
            else:
                self._objects[tid].region    = d["region"]
                self._objects[tid].last_seen = now

        gone = [t for t, m in self._objects.items()
                if t not in active and now - m.last_seen > self.GRACE_PERIOD]
        for t in gone:
            del self._objects[t]
        self._prev_center_ids -= set(gone)

    def _sensor_ok(self, fused) -> bool:
        if fused is None or fused.sensor_cm is None:
            return True
        return fused.sensor_cm >= CLEAR_SENSOR_CM

    def _was_busy(self, now) -> bool:
        return (now - self._last_urgent  < 30.0
                or now - self._last_warning < 30.0)

    def _route(self, state) -> str:
        ls  = self._score(state.zones["left"])
        rs  = self._score(state.zones["right"])
        obj = state.closest_class or "obstacle"
        if ls == 0 and rs == 0:
            side = ("left" if state.zone_counts["left"]
                    <= state.zone_counts["right"] else "right")
            return f"{obj} ahead, move {side}"
        if ls < rs: return f"{obj} ahead, move left"
        if rs < ls: return f"{obj} ahead, move right"
        return f"{obj} ahead, stop and wait"

    @staticmethod
    def _score(s): return {"clear": 0, "occupied": 1, "crowded": 2}.get(s, 1)