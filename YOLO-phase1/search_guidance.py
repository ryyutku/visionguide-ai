# search_guidance.py
# State machine: scanning → guiding → close → found → picked_up

import time
import logging

log = logging.getLogger("search")

SCANNING  = "scanning"
GUIDING   = "guiding"
CLOSE     = "close"
FOUND     = "found"
PICKED_UP = "picked_up"
IDLE      = "idle"


class SearchGuidance:
    def __init__(self, target: str):
        self.target  = target
        self._state  = SCANNING

        self._cooldowns = {
            "scanning":   6.0,
            "direction":  3.0,
            "close":      2.5,
            "found":      0.0,
        }
        self._last_said:      dict[str, float] = {}
        self._last_region     = None
        self._last_proximity  = None
        self._last_vertical   = None
        self._found_announced = False

        # Track consecutive frames where target disappears AFTER found state
        # — used to auto-detect pickup
        self._frames_missing_after_found = 0
        self._PICKUP_MISSING_FRAMES      = 8

    def process(self, detections: list, speech) -> tuple[str, str]:
        now = time.time()

        if self._state in (PICKED_UP, IDLE):
            return self._state, ""

        targets = [d for d in detections if d["is_target"] and d["confirmed"]]

        # ── Target not visible ───────────────────────────────────────────
        if not targets:
            # Auto-detect pickup: if we were in FOUND and target disappears
            if self._state == FOUND:
                self._frames_missing_after_found += 1
                if self._frames_missing_after_found >= self._PICKUP_MISSING_FRAMES:
                    return self._handle_picked_up(speech)
                return FOUND, ""

            # Lost the target mid-search
            if self._state in (GUIDING, CLOSE):
                self._state          = SCANNING
                self._last_region    = None
                self._last_proximity = None
                self._found_announced = False
                msg = f"Lost {self.target}. Scan slowly left and right."
                speech.say(msg)
                log.info("LOST      %s", msg)
                self._last_said["scanning"] = now
                return SCANNING, msg

            # Still scanning — periodic prompt
            if self._allow("scanning", now):
                msg = f"Still looking for {self.target}."
                speech.say(msg)
                log.info("SCANNING  %s", msg)
                return SCANNING, msg

            return SCANNING, ""

        self._frames_missing_after_found = 0

        # ── Pick best detection ──────────────────────────────────────────
        best      = max(targets, key=lambda d: d["area_ratio"])
        region    = best["region"]
        proximity = best["proximity"]
        vertical  = best["vertical"]

        # ── FOUND — reachable and centred ────────────────────────────────
        if proximity == "reachable" and region == "center":
            self._state = FOUND
            if not self._found_announced:
                self._found_announced = True
                msg = self._found_message(vertical)
                speech.say_urgent(msg)
                log.info("FOUND     %s", msg)
                return FOUND, msg
            return FOUND, ""

        self._found_announced = False

        # ── Build directional message ────────────────────────────────────
        changed = (region   != self._last_region   or
                   proximity != self._last_proximity or
                   vertical  != self._last_vertical)

        self._last_region    = region
        self._last_proximity = proximity
        self._last_vertical  = vertical

        msg = self._build_direction_message(region, proximity, vertical)

        if changed:
            self._state = CLOSE if proximity == "near" else GUIDING
            if proximity in ("near", "reachable"):
                speech.say_urgent(msg)
            else:
                speech.say(msg)
            self._last_said["direction"] = now
            log.info("%-9s %s", self._state.upper(), msg)
            return self._state, msg

        # Repeat on cooldown
        key = "close" if proximity == "near" else "direction"
        if self._allow(key, now):
            speech.say(msg)
            return self._state, msg

        return self._state, ""

    def handle_got_it(self, speech) -> tuple[str, str]:
        """Call when user says 'got it' or presses the confirm button."""
        return self._handle_picked_up(speech)

    def reset(self, new_target: str):
        self.target               = new_target
        self._state               = SCANNING
        self._last_region         = None
        self._last_proximity      = None
        self._last_vertical       = None
        self._found_announced     = False
        self._frames_missing_after_found = 0
        self._last_said.clear()

    # ------------------------------------------------------------------

    def _handle_picked_up(self, speech) -> tuple[str, str]:
        self._state = PICKED_UP
        msg = f"Great, you have the {self.target}."
        speech.say_urgent(msg)
        log.info("PICKED_UP %s", msg)
        return PICKED_UP, msg

    def _found_message(self, vertical: str) -> str:
        height_hint = {
            "high": "above you — reach up.",
            "low":  "below you — reach down.",
            "mid":  "in front of you — reach out now.",
        }.get(vertical, "in front of you.")
        return f"{self.target} is {height_hint}"

    def _build_direction_message(self,
                                  region: str,
                                  proximity: str,
                                  vertical: str) -> str:
        # Distance hint
        dist = {
            "far":        "Look around. ",
            "near":       "Getting close. ",
            "reachable":  "",
        }.get(proximity, "")

        # Horizontal direction
        horiz = {
            "left":   "Turn left.",
            "right":  "Turn right.",
            "center": "Move forward slowly.",
        }.get(region, "Adjust position.")

        # Vertical hint — only add when object is clearly high or low
        vert_hint = ""
        if vertical == "high" and proximity != "far":
            vert_hint = " It is above you."
        elif vertical == "low" and proximity != "far":
            vert_hint = " Look down."

        return f"{dist}{horiz}{vert_hint}"

    def _allow(self, key: str, now: float) -> bool:
        cooldown = self._cooldowns.get(key, 3.0)
        last     = self._last_said.get(key, 0.0)
        if now - last >= cooldown:
            self._last_said[key] = now
            return True
        return False