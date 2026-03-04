# search_main.py

import cv2
from SearchDetector import SearchDetector
from search_mode import SearchMode

# Change this to whatever object you want
TARGET_OBJECT = "cup"

# Initialize detector
detector = SearchDetector("yolov8m.pt")

# Initialize search mode
search = SearchMode(TARGET_OBJECT)

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Get processed frame + detections
    processed_frame, detections = detector.get_detections(frame)

    # Run search logic
    search.process(detections, processed_frame.shape[1])

    # Show video with bounding boxes
    cv2.imshow("Search Mode", processed_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()