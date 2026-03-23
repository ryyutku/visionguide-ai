# guidance.py

import time
import logging
from scene import SceneState

log = logging.getLogger("guidance")

PRIORITY_HIGH   = 3
PRIORITY_MEDIUM = 2
PRIORITY_LOW    = 1


class GuidanceEngine:
    def __init__(self):
        self._cooldowns = {
            "urgent":  4.0,
            "warning": 5.0,
            "clear":   8.0,
        }
        self._last_said: dict[str, float] = {}

        self._prev_center           = "clear"
        self._cleared_announced     = False

    def decide(self, state: SceneState, speech=None) -> tuple[str | None, int]:
        """
        speech: optional SpeechEngine reference.
        When passed, center alerts call say_urgent() which interrupts
        any currently-playing side warning mid-sentence.
        """
        now    = time.time()
        center = state.zones["center"]
        center_blocked = center in ("occupied", "crowded")

        # ── 1. CENTER JUST BECAME BLOCKED ────────────────────────────────
        if center_blocked and self._prev_center == "clear":
            msg = self._route_around(state)
            self._prev_center       = center
            self._cleared_announced = False
            self._last_said["urgent"] = now

            log.info("URGENT  %s  [new obstruction]", msg)

            # Interrupt any side-warning that may be speaking right now
            if speech:
                speech.say_urgent(msg)
                return msg, PRIORITY_HIGH   # guidance returns it for UI banner too
            return msg, PRIORITY_HIGH

        # ── 2. CENTER STILL BLOCKED — repeat on slow interval ────────────
        if center_blocked:
            self._prev_center       = center
            self._cleared_announced = False
            msg = self._route_around(state)
            if self._allow("urgent", now):
                log.info("URGENT  %s  [repeat]", msg)
                if speech:
                    speech.say_urgent(msg)
                return msg, PRIORITY_HIGH
            return None, 0

        # ── 3. CENTER JUST CLEARED ───────────────────────────────────────
        if not center_blocked and self._prev_center != "clear":
            self._prev_center = "clear"
            if not self._cleared_announced:
                self._cleared_announced = True
                self._last_said["clear"] = now
                log.info("CLEARED Path clear, move forward")
                return "Path clear, move forward", PRIORITY_LOW

        self._prev_center = "clear"

        # ── 4. CLOSE OBJECT ON SIDE ──────────────────────────────────────
        if (state.closest_proximity == "close"
                and state.closest_region in ("left", "right")):
            side  = state.closest_region
            other = "right" if side == "left" else "left"
            msg   = f"{state.closest_class} close on your {side}, move {other}"
            if self._allow("warning", now):
                log.info("WARNING %s", msg)
                return msg, PRIORITY_MEDIUM
            return None, 0

        # ── 5. Periodic clear ────────────────────────────────────────────
        if self._allow("clear", now):
            log.info("CLEAR   Path clear")
            return "Path clear", PRIORITY_LOW

        return None, 0

    # ------------------------------------------------------------------

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

    def _allow(self, key: str, now: float) -> bool:
        cooldown = self._cooldowns[key]
        last     = self._last_said.get(key, 0.0)
        if now - last >= cooldown:
            self._last_said[key] = now
            return True
        return False