from ultralytics import YOLO
import cv2


class SearchDetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def get_detections(self, frame):

        # Run YOLO in prediction mode (NOT tracking)
        results = self.model.predict(frame, conf=0.3, verbose=False)

        height, width, _ = frame.shape
        left_boundary = width / 3
        right_boundary = 2 * width / 3

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                center_x = (x1 + x2) / 2
                area = (x2 - x1) * (y2 - y1)

                # Determine region
                if center_x < left_boundary:
                    region_name = "left"
                elif center_x < right_boundary:
                    region_name = "center"
                else:
                    region_name = "right"

                # Store ALL detections
                detections.append({
                    "class": class_name,
                    "area": area,
                    "region": region_name
                })

                # Draw box
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                label = f"{class_name} | {region_name}"

                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Draw region divider lines
        cv2.line(frame, (int(left_boundary), 0),
                 (int(left_boundary), height), (255, 0, 0), 2)

        cv2.line(frame, (int(right_boundary), 0),
                 (int(right_boundary), height), (255, 0, 0), 2)

        return frame, detections