from ultralytics import YOLO
import cv2

class DetectorTracker:
    def __init__(self, model_path):
        self.model  = YOLO(model_path)

    def get_detections(self, frame):
        # we usually do ret, frame = cap.read from the camera we read the whether the camera works and the get the frame
        results = self.model.track(frame, persist=True, verbose=True) # Running Yolo on the model
        # Extract height, width, from the frame
        _, width, _  = frame.shape

        left_boundary = width / 3
        right_boundary = 2 * width / 3

        #Prepare region containers for closest object per region
        regions = {
            "left":None,
            "center": None,
            "right": None
        }

        for result in results:
            boxes = result.boxes

            # Sometimes box id's can be null
            if boxes is None:
                continue

            for box in boxes:
                # Sometimes box id can be None
                if box.id is None:
                    continue

                # Convert tensor in to cordinate floats for safer calculations
                x1, y1, x2, y2 = box.xyxy[0].tolist() # Getting dimensions of the box

                # Calculating the area of the box, we use this to determine how close the object is to the main camera
                center_x = (x1 + x2)/ 2
                area = (x2 - x1) * (y2 - y1)

                # Track the id of the box
                track_id = int(box.id[0].item()) # if the box doesnt have an id we skip over it
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                # Determining the region
                if center_x < left_boundary:
                    region_name = "left"
                elif center_x < right_boundary:
                    region_name = "center"
                else:
                    region_name = "right"

                # Storing the closest object per region
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

                    # --- DRAWING SECTION (FOR DEBUGGING) ---

                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2
                    )

                    # Label text
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

                    # Draw vertical region divider lines
                cv2.line(
                    frame,
                    (int(left_boundary), 0),
                    (int(left_boundary), frame.shape[0]),
                    (255, 0, 0),
                    2
                )

                cv2.line(
                    frame,
                    (int(right_boundary), 0),
                    (int(right_boundary), frame.shape[0]),
                    (255, 0, 0),
                    2
                )
            return regions


# yolov8s.pt   (small)
# yolov8m.pt   (medium)
detector = DetectorTracker("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    regions = detector.get_detections(frame)

    # Print closest objects per region
    print(regions)

    cv2.imshow("Navigation Debug View", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()