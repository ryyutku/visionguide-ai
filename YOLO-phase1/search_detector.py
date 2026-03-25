# search_detector.py
# Uses yolov8n (fast) + frame skipping to maintain smooth video.
# Detection runs every N frames; skipped frames reuse last known positions.

import cv2
from ultralytics import YOLO

CONFIRM_FRAMES  = 3
DETECT_EVERY    = 2      # run YOLO every 2nd frame, reuse detections on odd frames


class SearchDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model         = YOLO(model_path)
        self._seen:   dict[str, int]   = {}
        self._last_detections: list    = []
        self._frame_count: int         = 0

    def get_detections(self, frame, target: str):
        frame      = cv2.resize(frame, (640, 480))
        frame      = cv2.flip(frame, 1)
        h, w       = frame.shape[:2]
        frame_area = w * h
        l_bound    = w / 3
        r_bound    = 2 * w / 3

        self._frame_count += 1
        run_detection = (self._frame_count % DETECT_EVERY == 0)

        if run_detection:
            results    = self.model.predict(frame, conf=0.35, verbose=False)
            detections = []
            active_cls = set()

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id   = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    conf       = float(box.conf[0])
                    is_target  = (class_name == target)

                    cx         = (x1 + x2) / 2
                    cy         = (y1 + y2) / 2
                    area       = (x2 - x1) * (y2 - y1)
                    area_ratio = area / frame_area

                    region = (
                        "left"   if cx < l_bound  else
                        "right"  if cx >= r_bound else
                        "center"
                    )

                    # Vertical zone: top third = high, bottom third = low, else mid
                    if cy < h / 3:
                        vertical = "high"
                    elif cy > 2 * h / 3:
                        vertical = "low"
                    else:
                        vertical = "mid"

                    proximity = (
                        "reachable" if area_ratio > 0.15 else
                        "near"      if area_ratio > 0.05 else
                        "far"
                    )

                    if is_target:
                        self._seen[class_name] = self._seen.get(class_name, 0) + 1
                        active_cls.add(class_name)
                        confirmed = self._seen[class_name] >= CONFIRM_FRAMES
                    else:
                        confirmed = True

                    detections.append({
                        "class":      class_name,
                        "region":     region,
                        "vertical":   vertical,
                        "area_ratio": area_ratio,
                        "proximity":  proximity,
                        "conf":       conf,
                        "confirmed":  confirmed,
                        "is_target":  is_target,
                        "bbox":       (int(x1), int(y1), int(x2), int(y2)),
                    })

            # Clean stale tracking
            stale = set(self._seen.keys()) - active_cls
            for k in stale:
                self._seen.pop(k, None)

            self._last_detections = detections
        else:
            # Reuse last frame's detections — still draw them
            detections = self._last_detections

        # Draw
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            if d["is_target"]:
                color = (0, 220, 80) if d["confirmed"] else (80, 150, 80)
                lbl   = f"{d['class']} | {d['region']} | {d['proximity']}"
            else:
                color = (60, 60, 60)
                lbl   = d["class"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, lbl, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        cv2.line(frame, (int(l_bound), 0), (int(l_bound), h), (40, 40, 40), 1)
        cv2.line(frame, (int(r_bound), 0), (int(r_bound), h), (40, 40, 40), 1)

        return frame, detections