# main.py
#
# Laptop:          python main.py
# Raspberry Pi:    CAMERA_BACKEND=picamera2 python main.py
# Force stub sensor (no hardware):  ULTRASONIC_BACKEND=stub python main.py
#
# Files needed:   detector.py  scene.py  speech.py  guidance.py
#                 ultrasonic.py  sensor_fusion.py

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

from detector      import DetectorTracker
from scene         import SceneAnalyzer
from guidance      import GuidanceEngine
from speech        import SpeechEngine, PRIORITY_HIGH
from ultrasonic    import UltrasonicSensor
from sensor_fusion import SensorFusion


def open_camera():
    """
    Returns a camera object with a .read() → (bool, frame) method.
    Laptop:        cv2.VideoCapture (default)
    Raspberry Pi:  set CAMERA_BACKEND=picamera2
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
                return True, frame[:, :, ::-1]   # RGB → BGR

            def set(self, *a):
                pass

            def release(self):
                self._cam.stop()

        log.info("Camera: picamera2")
        return PiCamWrapper()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    log.info("Camera: OpenCV VideoCapture")
    return cap


def main():
    detector = DetectorTracker("yolov8n.pt")
    scene    = SceneAnalyzer()
    guidance = GuidanceEngine()
    speech   = SpeechEngine()
    sensor   = UltrasonicSensor()   # auto-selects GPIO on Pi, stub on laptop
    fusion   = SensorFusion()
    cap      = open_camera()

    log.info("System ready — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Vision ───────────────────────────────────────────────────────
        processed_frame, detections = detector.get_detections(frame)
        scene_state                 = scene.analyze(detections, frame.shape[1])

        # ── Sensor fusion ────────────────────────────────────────────────
        dist_cm = sensor.read_distance_cm()
        fused   = fusion.fuse(dist_cm, scene_state)

        # ── Guidance ─────────────────────────────────────────────────────
        message, priority = guidance.decide(
            scene_state, detections, speech, fused
        )

        if message:
            log.info("[ALERT p%d] %s", priority, message)
            if priority < PRIORITY_HIGH:
                speech.say(message, priority)

        _draw_hud(processed_frame, scene_state, message, fused)

        cv2.imshow("VisionGuide", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    log.info("Shutting down.")
    speech.shutdown()
    sensor.close()
    cap.release()
    cv2.destroyAllWindows()


def _draw_hud(frame, scene_state, message, fused=None):
    h, w = frame.shape[:2]

    zone_colors = {
        "clear":    (50,  200,  50),
        "occupied": (240, 140,  30),
        "crowded":  (60,   60, 220),
    }

    positions = {"left": 16, "center": w // 2 - 30, "right": w - 84}
    for zone, x in positions.items():
        status = scene_state.zones[zone]
        color  = zone_colors.get(status, (180, 180, 180))
        cv2.putText(frame, f"{zone.upper()}: {status}", (x, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    if message:
        cv2.putText(frame, message, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2, cv2.LINE_AA)

    # Sensor overlay — shown in top-right corner
    if fused and fused.sensor_cm is not None:
        cm    = fused.sensor_cm
        band  = fused.proximity
        color = {
            "critical": (0,   0,   255),
            "close":    (0,  100,  255),
            "medium":   (0,  200,  200),
            "far":      (80, 180,   80),
        }.get(band, (180, 180, 180))

        src_tag = "" if fused.source == "camera" else f" [{fused.source}]"
        label   = f"Sensor: {cm:.0f}cm {band}{src_tag}"
        cv2.putText(frame, label, (w - 230, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        # Sensor floor indicator
        if fused.sensor_floor_active:
            cv2.putText(frame, "! SENSOR FLOOR", (w - 180, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()