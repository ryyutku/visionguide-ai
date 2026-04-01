# detector.py

import cv2
from ultralytics import YOLO
from collections import defaultdict

# Only classes relevant to pedestrian / indoor navigation
IMPORTANT_CLASSES = {
    "person", "car", "bicycle", "motorcycle",
    "bus", "truck", "dog", "chair", "dining table",
    "couch", "potted plant", "bed", "toilet", "door"
}

# Minimum consecutive frames an object must appear before we report it.
# Filters out YOLO flicker / false positives.
CONFIRM_FRAMES = 3


class DetectorTracker:
    def __init__(self, model_path: str):
        self.model  = YOLO(model_path)
        self._smooth: dict[int, tuple] = {}   # track_id → smoothed bbox
        self._seen:   dict[int, int]   = defaultdict(int)  # track_id → frame count
        self.alpha = 0.5   # EMA weight for previous position (0=no smoothing, 1=frozen)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_detections(self, frame: "np.ndarray"):
        """
        Returns (annotated_frame, detections).

        Each detection dict:
            id       : int   — stable track id
            class    : str   — YOLO class name
            region   : str   — "left" | "center" | "right"
            area     : float — bounding-box pixel area (for proximity estimate)
            proximity: str   — "close" | "medium" | "far"
            confirmed: bool  — True once seen for CONFIRM_FRAMES consecutive frames
        """
        frame  = cv2.resize(frame, (640, 480))
        frame  = cv2.flip(frame, 1)   # mirror so user's left = left on screen
        h, w   = frame.shape[:2]
        l_bound = w / 3
        r_bound = 2 * w / 3
        frame_area = w * h          # 307 200 px for 640×480

        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=0.4,
            iou=0.5,
        )

        detections        = []
        active_ids: set   = set()

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                if box.id is None:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                track_id  = int(box.id.item())
                class_id  = int(box.cls[0])
                class_name = self.model.names[class_id]

                if class_name not in IMPORTANT_CLASSES:
                    continue

                # Smooth bbox
                x1, y1, x2, y2 = self._smooth_bbox(track_id, x1, y1, x2, y2)

                # Increment confirmation counter
                self._seen[track_id] += 1
                active_ids.add(track_id)

                center_x = (x1 + x2) / 2
                area     = (x2 - x1) * (y2 - y1)

                region = (
                    "left"   if center_x < l_bound  else
                    "right"  if center_x >= r_bound  else
                    "center"
                )

                # Proximity based on fraction of frame the box occupies.
                # These thresholds are relative so they work for any object size.
                area_ratio = area / frame_area
                proximity  = (
                    "close"  if area_ratio > 0.18 else
                    "medium" if area_ratio > 0.05 else
                    "far"
                )

                confirmed = self._seen[track_id] >= CONFIRM_FRAMES

                detections.append({
                    "id":        track_id,
                    "class":     class_name,
                    "region":    region,
                    "area":      area,
                    "proximity": proximity,
                    "confirmed": confirmed,
                })

                # Annotate frame
                color = (0, 255, 0) if confirmed else (180, 180, 180)
                cv2.rectangle(frame,
                              (int(x1), int(y1)), (int(x2), int(y2)),
                              color, 2)
                label = f"{class_name} [{region}] {proximity}"
                if not confirmed:
                    label += " ?"
                cv2.putText(frame, label,
                            (int(x1), max(int(y1) - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Clean up objects that have left the frame
        gone = set(self._seen.keys()) - active_ids
        for tid in gone:
            self._seen.pop(tid, None)
            self._smooth.pop(tid, None)

        # Draw zone dividers
        cv2.line(frame, (int(l_bound), 0), (int(l_bound), h), (200, 200, 200), 1)
        cv2.line(frame, (int(r_bound), 0), (int(r_bound), h), (200, 200, 200), 1)

        return frame, detections

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _smooth_bbox(self, track_id, x1, y1, x2, y2):
        """Exponential moving average to reduce bbox jitter."""
        if track_id not in self._smooth:
            self._smooth[track_id] = (x1, y1, x2, y2)
            return x1, y1, x2, y2

        px1, py1, px2, py2 = self._smooth[track_id]
        a = self.alpha
        sx1 = a * px1 + (1 - a) * x1
        sy1 = a * py1 + (1 - a) * y1
        sx2 = a * px2 + (1 - a) * x2
        sy2 = a * py2 + (1 - a) * y2
        self._smooth[track_id] = (sx1, sy1, sx2, sy2)
        return sx1, sy1, sx2, sy2