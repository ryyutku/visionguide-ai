# scene.py

from dataclasses import dataclass, field
from typing import Literal

ZoneStatus     = Literal["clear", "occupied", "crowded"]
ProximityLevel = Literal["close", "medium", "far", "none"]


@dataclass
class SceneState:
    zones: dict = field(default_factory=lambda: {
        "left": "clear", "center": "clear", "right": "clear",
    })
    closest_proximity: ProximityLevel = "none"
    closest_class:     str            = ""
    closest_region:    str            = ""
    zone_counts: dict = field(default_factory=lambda: {
        "left": 0, "center": 0, "right": 0,
    })
    center_clear: bool = True


class SceneAnalyzer:
    CROWDED_THRESHOLD = 3
    _PROX_RANK = {"close": 3, "medium": 2, "far": 1, "none": 0}

    def analyze(self, detections: list, frame_width: int) -> SceneState:
        state     = SceneState()
        confirmed = [d for d in detections if d["confirmed"]]

        for d in confirmed:
            state.zone_counts[d["region"]] += 1

        for zone, count in state.zone_counts.items():
            if count == 0:
                state.zones[zone] = "clear"
            elif count < self.CROWDED_THRESHOLD:
                state.zones[zone] = "occupied"
            else:
                state.zones[zone] = "crowded"

        state.center_clear = state.zones["center"] == "clear"

        best_rank = 0
        for d in confirmed:
            rank = self._PROX_RANK[d["proximity"]]
            if rank > best_rank:
                best_rank               = rank
                state.closest_proximity = d["proximity"]
                state.closest_class     = d["class"]
                state.closest_region    = d["region"]

        return state
