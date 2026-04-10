# run.py  —  Single launcher for the full VisionGuide system
#
# Starts both the navigation loop and the web dashboard server
# as separate threads in the same process, so they share memory
# through shared_state.py.
#
# Usage:
#   python run.py                  # default camera (index 0)
#   python run.py --camera 1       # if USB camera is on /dev/video1
#   python run.py --port 8080      # use different port (default 5000)
#   python run.py --show           # also open local cv2 window

import threading
import logging
import argparse
import os
import sys
import time

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


def get_local_ip() -> str:
    """Return the Pi's WiFi IP address for display."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


def run_navigation(camera_index: int, show: bool):
    """Runs the camera + YOLO + guidance pipeline."""
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
        log.error("Could not open camera %d", camera_index)
        return

    log.info("Navigation loop started (camera %d)", camera_index)

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


def run_server(port: int):
    """Runs the Flask dashboard server."""
    from server import app
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False,
            use_reloader=False)


def main():
    parser = argparse.ArgumentParser(description="VisionGuide — full system launcher")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port",   type=int, default=5000)
    parser.add_argument("--show",   action="store_true",
                        help="Also open local cv2 window")
    args = parser.parse_args()

    ip = get_local_ip()
    log.info("=" * 52)
    log.info("  VisionGuide starting")
    log.info("  Dashboard → http://%s:%d", ip, args.port)
    log.info("  Camera index: %d", args.camera)
    log.info("=" * 52)

    # Start Flask server in background thread
    server_thread = threading.Thread(
        target=run_server,
        args=(args.port,),
        daemon=True,
        name="server",
    )
    server_thread.start()
    log.info("Dashboard server started")
    time.sleep(1.0)   # give Flask a moment to bind the port

    # Run navigation loop in the main thread
    # (keeps keyboard interrupt / signals working normally)
    try:
        run_navigation(args.camera, args.show)
    except KeyboardInterrupt:
        log.info("Stopped by user.")
    finally:
        log.info("VisionGuide shut down.")


if __name__ == "__main__":
    main()
