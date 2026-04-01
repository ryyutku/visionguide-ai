# guidance.py  —  Smart navigation guidance with per-object state tracking
#
# v3 fixes:
#   • Minimum re-announce gap per object prevents rapid-fire alerts when a
#     close bounding box wobbles across zone boundaries frame-to-frame
#   • Grace period before pruning lost objects — a track ID that briefly
#     disappears (YOLO flicker on very close objects) is not treated as new
#   • "Path clear" is suppressed if the same object was in center very recently
#     (stops the blocked→clear→blocked loop caused by bbox jitter)

import time
import logging
from dataclasses import dataclass, field
from scene import SceneState

log = logging.getLogger("guidance")

PRIORITY_HIGH   = 3
PRIORITY_MEDIUM = 2
PRIORITY_LOW    = 1

_PROX_RANK = {"far": 1, "medium": 2, "close": 3, "none": 0}


@dataclass
class _ObjectMemory:
    track_id:           int
    obj_class:          str
    region:             str
    last_proximity:     str   = "far"
    announced_at:       float = 0.0
    announcement_count: int   = 0
    first_seen:         float = field(default_factory=time.time)
    last_seen:          float = field(default_factory=time.time)  # for grace period


class GuidanceEngine:
    # ── Cooldowns ────────────────────────────────────────────────────────────
    COOLDOWN_URGENT  = 6.0
    COOLDOWN_WARNING = 7.0
    COOLDOWN_CLEAR   = 10.0

    # Minimum time before we re-announce the SAME object for any reason.
    # This is the key fix — even if zone logic thinks it's "new", we won't
    # speak about it again within this window.
    MIN_REANNOUNCE_GAP = 5.0

    # How long a track ID is kept in memory after it stops being detected.
    # Prevents YOLO flicker on close objects from resetting the object as new.
    OBJECT_GRACE_PERIOD = 2.5   # seconds

    # Queue mode: object stationary in center beyond this time → slow reminders
    QUEUE_MODE_THRESHOLD    = 8.0
    QUEUE_REMINDER_INTERVAL = 20.0

    # After an urgent alert, suppress "path clear" for this long.
    # Stops the blocked→clear→blocked flicker loop.
    CLEAR_SUPPRESS_AFTER_URGENT = 3.0

    def __init__(self):
        self._objects: dict[int, _ObjectMemory] = {}

        self._last_urgent:  float = 0.0
        self._last_warning: float = 0.0
        self._last_clear:   float = 0.0

        self._was_blocked:     bool  = False
        self._block_start:     float = 0.0
        self._cleared_said:    bool  = False

        self._prev_center_ids: set[int] = set()

    # ── Public ───────────────────────────────────────────────────────────────

    def decide(self, state: SceneState, detections: list, speech=None
               ) -> tuple[str | None, int]:
        now = time.time()
        self._sync_memory(detections, now)

        center_blocked = state.zones["center"] in ("occupied", "crowded")

        if center_blocked:
            msg, pri = self._handle_center_blocked(state, detections, now, speech)
            if msg:
                return msg, pri
            return None, 0

        # Center just cleared
        if self._was_blocked and not center_blocked:
            self._was_blocked = False
            self._block_start = 0.0
            self._prev_center_ids = set()

            # Suppress "path clear" if we only just fired an urgent — it's
            # probably bbox jitter, not a real clearance
            time_since_urgent = now - self._last_urgent
            if not self._cleared_said and time_since_urgent > self.CLEAR_SUPPRESS_AFTER_URGENT:
                self._cleared_said = True
                self._last_clear   = now
                log.info("CLEAR  Path clear, move forward")
                return "Path clear, move forward", PRIORITY_LOW
            return None, 0

        self._was_blocked  = False
        self._cleared_said = False

        # Side warning
        msg, pri = self._handle_side_warning(state, detections, now)
        if msg:
            return msg, pri

        # Periodic clear — only if things were actually busy recently
        if (now - self._last_clear >= self.COOLDOWN_CLEAR
                and self._was_previously_busy(now)):
            self._last_clear = now
            return "Path clear", PRIORITY_LOW

        return None, 0

    # ── Internal ─────────────────────────────────────────────────────────────

    def _handle_center_blocked(self, state, detections, now, speech):
        center_ids = {
            d["id"] for d in detections
            if d["confirmed"] and d["region"] == "center"
        }

        # New IDs = in center now but NOT seen in center last frame
        # AND not announced very recently (grace period prevents flicker re-triggers)
        new_ids = set()
        for tid in center_ids - self._prev_center_ids:
            mem = self._objects.get(tid)
            if mem is None:
                continue
            time_since_announced = now - mem.announced_at
            if time_since_announced >= self.MIN_REANNOUNCE_GAP:
                new_ids.add(tid)

        if not self._was_blocked:
            self._was_blocked  = True
            self._block_start  = now
            self._cleared_said = False

        self._prev_center_ids = center_ids
        time_blocked = now - self._block_start

        # ── New object entered center ────────────────────────────────────
        if new_ids:
            msg = self._route_around(state)
            self._last_urgent = now
            for tid in new_ids:
                if tid in self._objects:
                    mem = self._objects[tid]
                    mem.announced_at        = now
                    mem.announcement_count += 1
            log.info("URGENT  %s  [new center object id=%s]", msg, new_ids)
            if speech:
                speech.say_urgent(msg)
            return msg, PRIORITY_HIGH

        # ── Proximity escalation ─────────────────────────────────────────
        escalated_id, escalated_obj = self._check_proximity_escalation(
            center_ids, detections
        )
        if escalated_id is not None:
            msg = f"{escalated_obj.obj_class} getting closer ahead, stop"
            self._last_urgent               = now
            escalated_obj.announced_at       = now
            escalated_obj.announcement_count += 1
            log.info("URGENT  %s  [escalation id=%d]", msg, escalated_id)
            if speech:
                speech.say_urgent(msg)
            return msg, PRIORITY_HIGH

        # ── Queue mode ───────────────────────────────────────────────────
        if time_blocked >= self.QUEUE_MODE_THRESHOLD:
            if now - self._last_urgent >= self.QUEUE_REMINDER_INTERVAL:
                obj_name = state.closest_class or "obstacle"
                msg = f"{obj_name} still ahead, wait or find alternate route"
                self._last_urgent = now
                log.info("URGENT  %s  [queue mode reminder]", msg)
                if speech:
                    speech.say_urgent(msg)
                return msg, PRIORITY_HIGH
            return None, 0

        # ── Normal repeat ────────────────────────────────────────────────
        if now - self._last_urgent >= self.COOLDOWN_URGENT:
            msg = self._route_around(state)
            self._last_urgent = now
            log.info("URGENT  %s  [repeat]", msg)
            if speech:
                speech.say_urgent(msg)
            return msg, PRIORITY_HIGH

        return None, 0

    def _handle_side_warning(self, state, detections, now):
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

                global_ok = now - self._last_warning >= self.COOLDOWN_WARNING

                if (just_escalated or first_alert) and global_ok:
                    side  = state.closest_region
                    other = "right" if side == "left" else "left"
                    msg   = f"{d['class']} close on your {side}, move {other}"
                    self._last_warning      = now
                    mem.last_proximity      = "close"
                    mem.announced_at        = now
                    mem.announcement_count += 1
                    log.info("WARNING  %s  [id=%d escalated=%s]",
                             msg, tid, just_escalated)
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

    def _check_proximity_escalation(self, center_ids, detections):
        for d in detections:
            if d["id"] not in center_ids or not d["confirmed"]:
                continue
            tid = d["id"]
            mem = self._objects.get(tid)
            if mem is None:
                continue
            prev_rank = _PROX_RANK.get(mem.last_proximity, 0)
            curr_rank = _PROX_RANK.get(d["proximity"], 0)
            if curr_rank > prev_rank:
                mem.last_proximity = d["proximity"]
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

        # Grace period: only prune objects that have been gone long enough.
        # This prevents a 1-2 frame YOLO dropout from resetting the object.
        to_delete = [
            tid for tid, mem in self._objects.items()
            if tid not in active_ids
            and now - mem.last_seen > self.OBJECT_GRACE_PERIOD
        ]
        for tid in to_delete:
            del self._objects[tid]
        self._prev_center_ids -= set(to_delete)

    def _was_previously_busy(self, now) -> bool:
        recently = 30.0
        return (now - self._last_urgent  < recently or
                now - self._last_warning < recently)

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