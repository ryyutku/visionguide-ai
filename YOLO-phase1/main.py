# main.py

import cv2
import logging

# Silence noisy third-party loggers BEFORE they are imported
for _noisy in ["comtypes", "comtypes.client", "comtypes.server",
               "PIL", "ultralytics", "torch", "urllib3",
               "pyttsx3", "pyttsx3.driver", "pyttsx3.drivers"]:
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

logging.basicConfig(
    level=logging.INFO,           # INFO = only meaningful events, no per-frame noise
    format="%(asctime)s  %(name)-10s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

# Imports happen AFTER logging is configured so third-party loggers are already silenced
from detector import DetectorTracker   # noqa: E402
from scene    import SceneAnalyzer     # noqa: E402
from guidance import GuidanceEngine    # noqa: E402
from speech   import SpeechEngine      # noqa: E402


def main():
    detector = DetectorTracker("yolov8n.pt")
    scene    = SceneAnalyzer()
    guidance = GuidanceEngine()
    speech   = SpeechEngine()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    log.info("System ready — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, detections = detector.get_detections(frame)
        scene_state                 = scene.analyze(detections, frame.shape[1])
        message, priority           = guidance.decide(scene_state)

        if message:
            log.info("[ALERT p%d] %s", priority, message)
            speech.say(message, priority)

        _draw_hud(processed_frame, scene_state, message)

        cv2.imshow("VisionGuide", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    log.info("Shutting down.")
    speech.shutdown()
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
        label  = f"{zone_key.upper()}: {status}"
        cv2.putText(frame, label, (x, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    if message:
        cv2.putText(frame, message, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2, cv2.LINE_AA)


if __name__ == "__main__":
    main()