# run.py  —  Single launcher for the full VisionGuide system
#
# Usage:
#   python run.py                 # default settings
#   python run.py --camera 1      # if USB camera is /dev/video1
#   python run.py --port 8080     # different port
#   python run.py --show          # also open local cv2 window

import threading
import logging
import argparse
import time
import socket

for _noisy in ["comtypes", "comtypes.client", "comtypes.server",
               "PIL", "ultralytics", "torch", "urllib3",
               "pyttsx3", "pyttsx3.driver", "pyttsx3.drivers"]:
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-10s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run")


def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


def run_navigation(camera_index, show):
    import cv2
    from detector      import DetectorTracker
    from scene         import SceneAnalyzer
    from guidance      import GuidanceEngine
    from speech        import SpeechEngine, PRIORITY_HIGH
    from ultrasonic    import UltrasonicSensor
    from sensor_fusion import SensorFusion
    import shared_state

    detector = DetectorTracker("yolov8n.pt")
    scene    = SceneAnalyzer()
    guidance = GuidanceEngine()
    speech   = SpeechEngine()
    sensor   = UltrasonicSensor()
    fusion   = SensorFusion()

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        log.error("Cannot open camera %d — try --camera 1", camera_index)
        return

    log.info("Camera %d ready — navigation loop started", camera_index)

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

        shared_state.update_frame(processed)
        shared_state.update_state(
            scene_state, detections, message, priority, fused
        )

        if show:
            cv2.imshow("VisionGuide", processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    speech.shutdown()
    sensor.close()
    cap.release()
    if show:
        cv2.destroyAllWindows()


def run_server(port):
    from server import app
    app.run(host="0.0.0.0", port=port, threaded=True,
            debug=False, use_reloader=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port",   type=int, default=5000)
    parser.add_argument("--show",   action="store_true")
    args = parser.parse_args()

    ip = get_ip()
    print()
    print("  ╔══════════════════════════════════════════╗")
    print(f"  ║  VisionGuide starting                    ║")
    print(f"  ║  Open in browser:                        ║")
    print(f"  ║  http://{ip}:{args.port}".ljust(45) + "║")
    print("  ╚══════════════════════════════════════════╝")
    print()

    server_thread = threading.Thread(
        target=run_server, args=(args.port,),
        daemon=True, name="flask"
    )
    server_thread.start()
    time.sleep(1.5)   # let Flask bind before camera starts
    log.info("Dashboard ready at http://%s:%d", ip, args.port)

    try:
        run_navigation(args.camera, args.show)
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    main()