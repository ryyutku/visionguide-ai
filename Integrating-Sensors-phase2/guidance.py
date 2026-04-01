# guidance.py  —  Smart navigation guidance with per-object state tracking
#
# Key behaviours:
#   • Tracks each object by track_id — no repeat for same stable object
#   • Proximity escalation: re-alerts only when far→medium→close
#   • Queue mode: stationary center object goes quiet after 8 s
#   • "Path clear" only fires after center has been continuously empty
#     for CLEAR_SUSTAIN_SECONDS — eliminates false clears from bbox flicker
#   • Grace period before pruning lost objects (handles YOLO flicker on
#     very close objects)
#   • Minimum re-announce gap per object (safety net for zone jitter)

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
    last_seen:          float = field(default_factory=time.time)


class GuidanceEngine:

    # ── Cooldowns ─────────────────────────────────────────────────────────
    COOLDOWN_URGENT  = 6.0
    COOLDOWN_WARNING = 7.0
    COOLDOWN_CLEAR   = 10.0

    # Center must stay continuously empty for this long before we say
    # "path clear".  Filters out bbox-flicker false clearances.
    CLEAR_SUSTAIN_SECONDS = 2.5

    # Minimum gap before re-announcing the same object for any reason
    MIN_REANNOUNCE_GAP = 5.0

    # Track IDs kept in memory this long after disappearing (handles YOLO
    # dropping a very close object for a frame or two)
    OBJECT_GRACE_PERIOD = 2.5

    # Queue mode
    QUEUE_MODE_THRESHOLD    = 8.0
    QUEUE_REMINDER_INTERVAL = 20.0

    def __init__(self):
        self._objects: dict[int, _ObjectMemory] = {}

        self._last_urgent:  float = 0.0
        self._last_warning: float = 0.0
        self._last_clear:   float = 0.0

        self._was_blocked:     bool  = False
        self._block_start:     float = 0.0
        self._cleared_said:    bool  = False

        self._prev_center_ids: set[int] = set()

        # Timestamp when center FIRST became empty after being blocked.
        # None means center is currently occupied (or we haven't started yet).
        self._center_empty_since: float | None = None

    # ── Public ────────────────────────────────────────────────────────────

    def decide(self, state: SceneState, detections: list, speech=None
               ) -> tuple[str | None, int]:

        now = time.time()
        self._sync_memory(detections, now)

        center_blocked = state.zones["center"] in ("occupied", "crowded")

        # ── CENTER BLOCKED ───────────────────────────────────────────────
        if center_blocked:
            # Reset the "empty since" timer the moment center is occupied
            self._center_empty_since = None
            self._cleared_said       = False

            msg, pri = self._handle_center_blocked(state, detections, now, speech)
            if msg:
                return msg, pri
            return None, 0

        # ── CENTER EMPTY ─────────────────────────────────────────────────
        # Start (or keep) the sustain timer
        if self._center_empty_since is None:
            self._center_empty_since = now   # center just became empty

        time_empty = now - self._center_empty_since

        # Only act on clearance once the center has been empty long enough
        if self._was_blocked and time_empty >= self.CLEAR_SUSTAIN_SECONDS:
            self._was_blocked     = False
            self._block_start     = 0.0
            self._prev_center_ids = set()

            if not self._cleared_said:
                self._cleared_said = True
                self._last_clear   = now
                log.info("CLEAR  Path clear, move forward  [sustained %.1fs]",
                         time_empty)
                return "Path clear, move forward", PRIORITY_LOW

        # If we were never blocked, or haven't sustained yet, just clear flags
        if not center_blocked:
            self._was_blocked = False

        # ── SIDE WARNING ─────────────────────────────────────────────────
        msg, pri = self._handle_side_warning(state, detections, now)
        if msg:
            return msg, pri

        # ── PERIODIC CLEAR ───────────────────────────────────────────────
        if (now - self._last_clear >= self.COOLDOWN_CLEAR
                and self._was_previously_busy(now)
                and time_empty >= self.CLEAR_SUSTAIN_SECONDS):
            self._last_clear = now
            return "Path clear", PRIORITY_LOW

        return None, 0

    # ── Internal ──────────────────────────────────────────────────────────

    def _handle_center_blocked(self, state, detections, now, speech):
        center_ids = {
            d["id"] for d in detections
            if d["confirmed"] and d["region"] == "center"
        }

        # New IDs = entered center AND not announced recently
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
            log.info("URGENT  %s  [new center object id=%s]", msg, new_ids)
            if speech:
                speech.say_urgent(msg)
            return msg, PRIORITY_HIGH

        # Proximity escalation
        eid, emem = self._check_proximity_escalation(center_ids, detections)
        if eid is not None:
            msg = f"{emem.obj_class} getting closer ahead, stop"
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
                obj_name = state.closest_class or "obstacle"
                msg = f"{obj_name} still ahead, wait or find alternate route"
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
                global_ok      = now - self._last_warning >= self.COOLDOWN_WARNING

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

        # Grace period — only prune after object has been gone long enough
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