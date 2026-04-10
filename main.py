# main.py  —  Navigation loop
#
# Runs the camera + YOLO + guidance pipeline.
# Pushes every frame and state update into shared_state.py so that
# server.py can stream them to the browser dashboard over WiFi.
#
# Start everything with:   python run.py
# Or manually:             python main.py  (in one terminal)
#                          python server.py (in another terminal)
#
# Flags:
#   --camera 1     use /dev/video1 instead of default /dev/video0
#   --show         also open a local cv2 window (useful on laptop)

import cv2
import logging
import argparse

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
import shared_state


def main():
    parser = argparse.ArgumentParser(description="VisionGuide navigation loop")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default 0)")
    parser.add_argument("--show", action="store_true",
                        help="Also open a local cv2 window")
    args = parser.parse_args()

    detector = DetectorTracker("yolov8n.pt")
    scene    = SceneAnalyzer()
    guidance = GuidanceEngine()
    speech   = SpeechEngine()
    sensor   = UltrasonicSensor()
    fusion   = SensorFusion()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        log.error("Could not open camera %d — try --camera 1", args.camera)
        return

    log.info("Camera %d ready", args.camera)
    log.info("Navigation loop running — dashboard available via server.py")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        processed, detections = detector.get_detections(frame)
        scene_state           = scene.analyze(detections, frame.shape[1])
        dist_cm               = sensor.read_distance_cm()
        fused                 = fusion.fuse(dist_cm, scene_state)
        message, priority     = guidance.decide(
            scene_state, detections, speech, fused
        )

        if message:
            log.info("[ALERT p%d] %s", priority, message)
            if priority < PRIORITY_HIGH:
                speech.say(message, priority)

        # Push to browser dashboard
        shared_state.update_frame(processed)
        shared_state.update_state(
            scene_state, detections, message, priority, fused
        )

        if args.show:
            cv2.imshow("VisionGuide", processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    log.info("Shutting down.")
    speech.shutdown()
    sensor.close()
    cap.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
