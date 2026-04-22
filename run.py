# run.py  —  VisionGuide launcher
#
# Usage:
#   python run.py                  # auto-detects camera
#   python run.py --camera 2       # force a specific camera index
#   python run.py --port 8080

import threading
import logging
import argparse
import time
import socket
import queue

for _noisy in ["comtypes", "comtypes.client", "comtypes.server",
               "PIL", "ultralytics", "torch", "urllib3",
               "pyttsx3", "pyttsx3.driver", "pyttsx3.drivers"]:
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(name)-10s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("run")


def get_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def find_camera(preferred: int = None) -> int:
    """
    Returns the first working camera index.
    If preferred is given, tries that first.
    Searches indices 0-9.
    """
    import cv2
    candidates = list(range(10))
    if preferred is not None:
        # Try preferred first, then the rest
        candidates = [preferred] + [i for i in candidates if i != preferred]

    for i in candidates:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                log.info("Camera found at index %d", i)
                return i
        cap.release()

    return -1   # not found


# ── Thread 1: Capture ─────────────────────────────────────────────────────────

def capture_loop(camera_index: int, frame_queue: queue.Queue,
                 stop: threading.Event):
    import cv2

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        log.error("Cannot open camera %d", camera_index)
        stop.set()
        return

    log.info("Capture ready (camera %d)", camera_index)

    while not stop.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # 🔄 FLIP THE FRAME 
        frame = cv2.flip(frame, -1)  # -1 = both horizontal and vertical (180° rotation)
        # Alternatives:
        # frame = cv2.flip(frame, 0)   # 0 = vertical flip only
        # frame = cv2.flip(frame, 1)   # 1 = horizontal flip only

        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)

    cap.release()
    log.info("Capture stopped")


# ── Thread 2: Inference ───────────────────────────────────────────────────────

def inference_loop(frame_queue: queue.Queue, stop: threading.Event):
    from detector      import DetectorTracker
    from scene         import SceneAnalyzer
    from guidance      import GuidanceEngine
    from speech        import SpeechEngine, PRIORITY_HIGH
    from ultrasonic    import UltrasonicSensor
    from sensor_fusion import SensorFusion
    from cloud_logger  import CloudLogger
    import shared_state

    detector = DetectorTracker()
    scene    = SceneAnalyzer()
    guidance = GuidanceEngine()
    speech   = SpeechEngine()
    sensor   = UltrasonicSensor()
    fusion   = SensorFusion()
    cloud    = CloudLogger()

    # ── Store last state for cloud command handlers ──────────────────────
    last_state = {
        "scene_state": None,
        "detections": [],
        "fused": None,
    }

    # ── Register cloud command handlers ──────────────────────────────────
    def handle_status(payload):
        """Remote command: immediately log current status to cloud."""
        log.info("Cloud command: STATUS received")
        # Use the most recent state captured
        scene_state = last_state["scene_state"]
        detections  = last_state["detections"]
        fused       = last_state["fused"]

        if scene_state is None:
            log.warning("STATUS command: no state available yet")
            return

        # Log an alert with "manual status" message
        cloud.log_alert(
            message           = "Manual status request (remote)",
            priority          = 1,
            zone_states       = scene_state.zones,
            closest_class     = scene_state.closest_class,
            closest_region    = scene_state.closest_region,
            closest_proximity = scene_state.closest_proximity,
        )

        # Also log a sensor reading
        cloud.log_sensor(
            sensor_cm       = fused.sensor_cm if fused else None,
            sensor_band     = fused.proximity if fused else "none",
            object_count    = len(detections),
            confirmed_count = sum(1 for d in detections if d["confirmed"]),
        )

        log.info("STATUS command: data sent to cloud")

    def handle_set_volume(payload):
        """Remote command: change speech volume (future enhancement)."""
        vol = payload.get("volume", 80)
        log.info("Cloud command: SET_VOLUME to %d%%", vol)
        # Could extend speech.py to support volume changes
        # speech.set_volume(vol)

    cloud.register_command_handler("STATUS", handle_status)
    cloud.register_command_handler("SET_VOLUME", handle_set_volume)

    log.info("Inference ready (cloud commands enabled)")

    frames_done = 0
    fps_clock   = time.time()
    sensor_tick = 0

    while not stop.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        processed, detections = detector.get_detections(frame)
        scene_state           = scene.analyze(detections, frame.shape[1])
        dist_cm               = sensor.read_distance_cm()
        fused                 = fusion.fuse(dist_cm, scene_state)
        message, priority     = guidance.decide(
            scene_state, detections, speech, fused
        )

        # Store for command handlers
        last_state["scene_state"] = scene_state
        last_state["detections"]  = detections
        last_state["fused"]       = fused

        if message:
            log.info("[ALERT p%d] %s", priority, message)
            if priority < PRIORITY_HIGH:
                speech.say(message, priority)
            cloud.log_alert(
                message           = message,
                priority          = priority,
                zone_states       = scene_state.zones,
                closest_class     = scene_state.closest_class,
                closest_region    = scene_state.closest_region,
                closest_proximity = scene_state.closest_proximity,
            )

        sensor_tick += 1
        if sensor_tick >= 10:
            sensor_tick = 0
            cloud.log_sensor(
                sensor_cm       = fused.sensor_cm if fused else None,
                sensor_band     = fused.proximity if fused else "none",
                object_count    = len(detections),
                confirmed_count = sum(1 for d in detections if d["confirmed"]),
            )

        shared_state.update_frame(processed)
        shared_state.update_state(
            scene_state, detections, message, priority, fused
        )

        frames_done += 1
        elapsed = time.time() - fps_clock
        if elapsed >= 5.0:
            log.info("Inference %.1f FPS", frames_done / elapsed)
            frames_done = 0
            fps_clock   = time.time()

    cloud.shutdown()
    speech.shutdown()
    sensor.close()
    log.info("Inference stopped")


# ── Thread 3: Flask ───────────────────────────────────────────────────────────

def server_loop(port: int):
    from server import app
    app.run(host="0.0.0.0", port=port, threaded=True,
            debug=False, use_reloader=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index (auto-detected if not specified)")
    parser.add_argument("--port",   type=int, default=5000)
    args = parser.parse_args()

    # Find camera before starting anything else
    log.info("Scanning for camera...")
    cam_index = find_camera(args.camera)
    if cam_index == -1:
        log.error("No working camera found.")
        log.error("Check that your USB camera is plugged in, then run:")
        log.error("  ls /dev/video*")
        log.error("  python run.py --camera <index>")
        return

    ip = get_ip()
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║  VisionGuide                                 ║")
    print(f"  ║  Dashboard → http://{ip}:{args.port}".ljust(49) + "║")
    print(f"  ║  Camera    → index {cam_index}".ljust(49) + "║")
    print("  ║  Ctrl+C to stop                              ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()

    stop    = threading.Event()
    frame_q = queue.Queue(maxsize=1)

    flask_t = threading.Thread(target=server_loop, args=(args.port,),
                               daemon=True, name="flask")
    flask_t.start()
    time.sleep(1.2)
    log.info("Dashboard → http://%s:%d", ip, args.port)

    cap_t = threading.Thread(target=capture_loop,
                             args=(cam_index, frame_q, stop),
                             daemon=True, name="capture")
    cap_t.start()

    inf_t = threading.Thread(target=inference_loop,
                             args=(frame_q, stop),
                             daemon=True, name="inference")
    inf_t.start()

    try:
        while not stop.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("Stopping...")
        stop.set()

    inf_t.join(timeout=5)
    cap_t.join(timeout=3)
    log.info("Stopped.")


if __name__ == "__main__":
    main()