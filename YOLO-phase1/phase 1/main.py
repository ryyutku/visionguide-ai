# # main.py

import cv2
from detector import DetectorTracker
from intelligence1 import IntelligenceEngine

detector = DetectorTracker("../yolov8n.pt")
brain = IntelligenceEngine()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame, detections = detector.get_detections(frame)

    brain.update(detections)

    cv2.imshow("Assistive Navigation", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# main.py
#--------------------------------------------------------------------------
# import cv2
# from detector import DetectorTracker
# from intelligence import IntelligenceEngine
#
# detector = DetectorTracker("yolov8n.pt")
# brain = IntelligenceEngine()
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     processed_frame, detections = detector.get_detections(frame)
#
#     # IMPORTANT: use process() instead of update()
#     brain.process(detections, frame.shape[1])
#
#     cv2.imshow("Assistive Navigation", processed_frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()