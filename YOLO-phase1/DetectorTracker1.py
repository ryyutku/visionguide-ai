from ultralytics import YOLO
import cv2

IMPORTANT_CLASSES = {
    "person", "car", "bicycle", "motorcycle",
    "bus", "truck", "dog"
}

class DetectorTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.object_memory = {}
        self.alpha = 0.6

    def smooth_bbox(self, track_id, x1, y1, x2, y2):
        if track_id not in self.object_memory:
            self.object_memory[track_id] = (x1, y1, x2, y2)
            return x1, y1, x2, y2

        px1, py1, px2, py2 = self.object_memory[track_id]

        sx1 = self.alpha * px1 + (1 - self.alpha) * x1
        sy1 = self.alpha * py1 + (1 - self.alpha) * y1
        sx2 = self.alpha * px2 + (1 - self.alpha) * x2
        sy2 = self.alpha * py2 + (1 - self.alpha) * y2

        self.object_memory[track_id] = (sx1, sy1, sx2, sy2)
        return sx1, sy1, sx2, sy2

    def get_detections(self, frame):

        # 🔥 Resize and RETURN this resized frame
        frame = cv2.resize(frame, (640, 480))
        height, width, _ = frame.shape

        left_boundary = width / 3
        right_boundary = 2 * width / 3

        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=0.4,
            iou=0.5
        )

        regions = {"left": None, "center": None, "right": None}

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                if box.id is None:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                track_id = int(box.id.item())
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                # To enable to detect only the important objects
                # if class_name not in IMPORTANT_CLASSES:
                #     continue

                x1, y1, x2, y2 = self.smooth_bbox(
                    track_id, x1, y1, x2, y2
                )

                center_x = (x1 + x2) / 2
                area = (x2 - x1) * (y2 - y1)

                if center_x < left_boundary:
                    region_name = "left"
                elif center_x < right_boundary:
                    region_name = "center"
                else:
                    region_name = "right"

                if (
                    regions[region_name] is None or
                    area > regions[region_name]["area"]
                ):
                    regions[region_name] = {
                        "id": track_id,
                        "class": class_name,
                        "area": area,
                        "region": region_name
                    }

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                label = f"{class_name} | ID:{track_id} | {region_name}"

                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # 🔥 Always draw region lines
        cv2.line(frame, (int(left_boundary), 0),
                 (int(left_boundary), height), (255, 0, 0), 2)

        cv2.line(frame, (int(right_boundary), 0),
                 (int(right_boundary), height), (255, 0, 0), 2)

        return frame, regions


# ---------------- MAIN ---------------- #

detector = DetectorTracker("yolov8n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_frame, regions = detector.get_detections(frame)

    print(regions)

    cv2.imshow("Navigation Debug View", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()