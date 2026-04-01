# main.py
#
# Run with:  python main.py
# On Pi with Pi Camera add:  CAMERA_BACKEND=picamera2 python main.py
#
# Ultrasonic sensor is NOT active yet.
# Search "TODO: ULTRASONIC" to find the integration points when ready.

import cv2
import logging
import os

for _noisy in ["comtypes", "comtypes.client", "comtypes.server",
               "PIL", "ultralytics", "torch", "urllib3",
               "pyttsx3", "pyttsx3.driver", "pyttsx3.drivers"]:
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-10s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

from detector import DetectorTracker
from scene    import SceneAnalyzer
from guidance import GuidanceEngine
from speech   import SpeechEngine, PRIORITY_HIGH

# TODO: ULTRASONIC — uncomment when sensor is ready
# from ultrasonic import UltrasonicSensor, DIST_CRITICAL


def open_camera():
    """
    Returns a camera object with a .read() method.
    Defaults to OpenCV webcam (laptop).
    Set CAMERA_BACKEND=picamera2 on Raspberry Pi.
    """
    backend = os.environ.get("CAMERA_BACKEND", "").lower()

    if backend == "picamera2":
        from picamera2 import Picamera2

        class PiCamWrapper:
            def __init__(self):
                self._cam = Picamera2()
                cfg = self._cam.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                self._cam.configure(cfg)
                self._cam.start()

            def read(self):
                frame = self._cam.capture_array()
                bgr   = frame[:, :, ::-1]   # RGB → BGR for OpenCV
                return True, bgr

            def set(self, *a):
                pass

            def release(self):
                self._cam.stop()

        log.info("Using picamera2 backend")
        return PiCamWrapper()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    log.info("Using OpenCV VideoCapture backend")
    return cap


def main():
    detector = DetectorTracker("yolov8n.pt")
    scene    = SceneAnalyzer()
    guidance = GuidanceEngine()
    speech   = SpeechEngine()
    cap      = open_camera()

    # TODO: ULTRASONIC — initialise sensor here
    # sensor = UltrasonicSensor()

    log.info("System ready — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # TODO: ULTRASONIC — read sensor and override if critical
        # dist_cm     = sensor.read_distance_cm()
        # sensor_band = sensor.proximity_band(dist_cm)
        # if sensor_band == "critical":
        #     speech.say_urgent(f"Stop! Object {int(dist_cm)} cm ahead")
        #     continue

        processed_frame, detections = detector.get_detections(frame)
        scene_state                 = scene.analyze(detections, frame.shape[1])
        message, priority           = guidance.decide(scene_state, detections, speech)

        if message:
            log.info("[ALERT p%d] %s", priority, message)
            if priority < PRIORITY_HIGH:
                speech.say(message, priority)

        _draw_hud(processed_frame, scene_state, message)

        cv2.imshow("VisionGuide", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    log.info("Shutting down.")
    speech.shutdown()
    # TODO: ULTRASONIC — sensor.close()
    cap.release()
    cv2.destroyAllWindows()


def _draw_hud(frame, scene_state, message):
    h, w = frame.shape[:2]

    zone_colors = {
        "clear":    (50,  200, 50),
        "occupied": (240, 140, 30),
        "crowded":  (60,  60,  220),
    }

    positions = {"left": 16, "center": w // 2 - 30, "right": w - 84}
    for zone_key, x in positions.items():
        status = scene_state.zones[zone_key]
        color  = zone_colors.get(status, (180, 180, 180))
        cv2.putText(frame, f"{zone_key.upper()}: {status}", (x, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    if message:
        cv2.putText(frame, message, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2, cv2.LINE_AA)


if __name__ == "__main__":
    main()