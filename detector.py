# detector.py

import cv2
import os
from ultralytics import YOLO
from collections import defaultdict

IMPORTANT_CLASSES = {
    "person", "car", "bicycle", "motorcycle",
    "bus", "truck", "dog", "chair", "dining table",
    "couch", "potted plant", "bed", "toilet", "door"
}

CONFIRM_FRAMES = 3
YOLO_INPUT_SIZE = 320


def _find_model() -> tuple[str, str]:
    """Return (model_path, task) preferring fastest available."""
    # Priority order:
    # 1. NCNN (fastest on Pi, uses ARM NEON)
    # 2. TFLite INT8 (quantized, needs tflite-runtime)
    # 3. PyTorch (slowest, fallback)

    ncnn_model = "yolov8n_ncnn_model"
    tflite_int8 = "yolov8n_int8.tflite"

    if os.path.exists(ncnn_model):
        print(f"[detector] Using NCNN model: {ncnn_model} (fast)")
        return ncnn_model, "detect"
    elif os.path.exists(tflite_int8):
        print(f"[detector] Using TFLite INT8: {tflite_int8} (quantized)")
        return tflite_int8, "detect"
    else:
        print("[detector] Using PyTorch model: yolov8n.pt")
        print("[detector] Tip: run 'python export_model.py' for 2-3x speedup")
        return "yolov8n.pt", "detect"


class DetectorTracker:
    def __init__(self, model_path: str = None):
        if model_path:
            path, task = model_path, "detect"
        else:
            path, task = _find_model()

        self.model = YOLO(path, task=task)
        self._smooth: dict[int, tuple] = {}
        self._seen: dict[int, int] = defaultdict(int)
        self.alpha = 0.5

    def get_detections(self, frame):
        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]

        l_bound = w / 3
        r_bound = 2 * w / 3
        frame_area = w * h

        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=0.40,
            iou=0.45,
            imgsz=YOLO_INPUT_SIZE,
        )

        detections: list = []
        active_ids: set = set()

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if box.id is None:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                track_id = int(box.id.item())
                class_name = self.model.names[int(box.cls[0])]

                if class_name not in IMPORTANT_CLASSES:
                    continue

                x1, y1, x2, y2 = self._smooth_bbox(track_id, x1, y1, x2, y2)
                self._seen[track_id] += 1
                active_ids.add(track_id)

                center_x = (x1 + x2) / 2
                area = (x2 - x1) * (y2 - y1)
                area_ratio = area / frame_area

                region = (
                    "left" if center_x < l_bound else
                    "right" if center_x >= r_bound else
                    "center"
                )
                proximity = (
                    "close" if area_ratio > 0.18 else
                    "medium" if area_ratio > 0.05 else
                    "far"
                )
                confirmed = self._seen[track_id] >= CONFIRM_FRAMES

                detections.append({
                    "id": track_id,
                    "class": class_name,
                    "region": region,
                    "area": area,
                    "proximity": proximity,
                    "confirmed": confirmed,
                })

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

        for tid in set(self._seen.keys()) - active_ids:
            self._seen.pop(tid, None)
            self._smooth.pop(tid, None)

        cv2.line(frame, (int(l_bound), 0), (int(l_bound), h), (200, 200, 200), 1)
        cv2.line(frame, (int(r_bound), 0), (int(r_bound), h), (200, 200, 200), 1)

        return frame, detections

    def _smooth_bbox(self, track_id, x1, y1, x2, y2):
        if track_id not in self._smooth:
            self._smooth[track_id] = (x1, y1, x2, y2)
            return x1, y1, x2, y2
        px1, py1, px2, py2 = self._smooth[track_id]
        a = self.alpha
        s = (a * px1 + (1 - a) * x1, a * py1 + (1 - a) * y1,
             a * px2 + (1 - a) * x2, a * py2 + (1 - a) * y2)
        self._smooth[track_id] = s
        return s