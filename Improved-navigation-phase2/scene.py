# scene.py
#
# Looks at ALL detections together and produces a simple scene description
# that the GuidanceEngine can reason about — instead of reasoning object by object.

from dataclasses import dataclass, field
from typing import Literal

ZoneStatus = Literal["clear", "occupied", "crowded"]
ProximityLevel = Literal["close", "medium", "far", "none"]


@dataclass
class SceneState:
    # Per-zone status
    zones: dict = field(default_factory=lambda: {
        "left":   "clear",
        "center": "clear",
        "right":  "clear",
    })

    # Closest confirmed object overall and its proximity
    closest_proximity: ProximityLevel = "none"
    closest_class:     str            = ""
    closest_region:    str            = ""

    # How many confirmed objects are in each zone
    zone_counts: dict = field(default_factory=lambda: {
        "left": 0, "center": 0, "right": 0
    })

    # Is the center fully clear of confirmed objects?
    center_clear: bool = True


class SceneAnalyzer:
    """
    Turns a flat list of per-object detections into a structured SceneState.
    Only confirmed objects (seen for CONFIRM_FRAMES+) count toward scene logic.
    """

    # How many confirmed objects in a zone before it is "crowded"
    CROWDED_THRESHOLD = 3

    # Proximity rank — higher = closer
    _PROX_RANK = {"close": 3, "medium": 2, "far": 1, "none": 0}

    def analyze(self, detections: list, frame_width: int) -> SceneState:
        state = SceneState()

        confirmed = [d for d in detections if d["confirmed"]]

        # Count objects per zone
        for d in confirmed:
            state.zone_counts[d["region"]] += 1

        # Assign zone status
        for zone, count in state.zone_counts.items():
            if count == 0:
                state.zones[zone] = "clear"
            elif count < self.CROWDED_THRESHOLD:
                state.zones[zone] = "occupied"
            else:
                state.zones[zone] = "crowded"

        state.center_clear = state.zones["center"] == "clear"

        # Find the closest confirmed object
        best_rank = 0
        for d in confirmed:
            rank = self._PROX_RANK[d["proximity"]]
            if rank > best_rank:
                best_rank             = rank
                state.closest_proximity = d["proximity"]
                state.closest_class     = d["class"]
                state.closest_region    = d["region"]

        return state
