# run.py  —  VisionGuide launcher
#
# Usage:
#   python3 run.py                 # auto-detects camera
#   python3 run.py --camera 2      # force camera index
#   python3 run.py --port 8080     # different port
#
# Architecture (3 threads):
#   CAPTURE   — reads camera frames, keeps only latest (drops stale)
#   INFERENCE — YOLO + guidance + sensor fusion + cloud logging
#   SERVER    — Flask MJPEG stream + JSON state (independent of YOLO speed)
#
# Memory note: all heavy imports (cv2, torch/YOLO, requests) are
# intentionally inside the thread functions, NOT at module level.
# This prevents the main process from pre-loading everything before
# threads start, which was causing OOM kills on the Pi.

import threading
import logging
import argparse
import time
import socket
import queue

# Only lightweight stdlib imports at module level
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
    import cv2   # imported here — not at module level
    candidates = list(range(10))
    if preferred is not None:
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
    return -1


# ── Thread 1: Capture ─────────────────────────────────────────────────────────

def capture_loop(camera_index: int, frame_queue: queue.Queue,
                 stop: threading.Event):
    import cv2   # imported inside thread — does not duplicate in memory on Linux (COW)

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

        # Flip 180° if camera is mounted upside down — remove if not needed
        frame = cv2.flip(frame, -1)

        # Keep only the latest frame — drop stale ones
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
    # All heavy imports inside this function
    import cv2
    import datetime
    import requests as req

    from detector      import DetectorTracker
    from scene         import SceneAnalyzer
    from guidance      import GuidanceEngine
    from speech        import SpeechEngine, PRIORITY_HIGH
    from ultrasonic    import UltrasonicSensor
    from sensor_fusion import SensorFusion
    from cloud_logger  import CloudLogger, SUPABASE_URL, SUPABASE_KEY
    import shared_state

    detector = DetectorTracker()
    scene    = SceneAnalyzer()
    guidance = GuidanceEngine()
    speech   = SpeechEngine()
    sensor   = UltrasonicSensor()
    fusion   = SensorFusion()
    cloud    = CloudLogger()

    # Shared state for command handlers
    last_state = {
        "scene_state": None,
        "detections":  [],
        "fused":       None,
        "frame":       None,
    }
    night_mode_enabled = False

    # ── Image upload helper ───────────────────────────────────────────────
    def upload_image(frame_bgr) -> str | None:
        """Encode frame as JPEG and upload to Supabase Storage. Returns public URL or None."""
        if not SUPABASE_URL or not SUPABASE_KEY:
            log.warning("Image upload skipped — no Supabase credentials")
            return None
        try:
            _, buf = cv2.imencode(".jpg", frame_bgr,
                                  [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_bytes = buf.tobytes()
            timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename    = f"capture_{timestamp}.jpg"
            url         = (f"{SUPABASE_URL}/storage/v1/object/"
                           f"visionguide-images/{filename}")
            headers = {
                "apikey":        SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type":  "image/jpeg",
            }
            r = req.post(url, data=image_bytes, headers=headers, timeout=10)
            if r.status_code in (200, 201):
                public = (f"{SUPABASE_URL}/storage/v1/object/public/"
                          f"visionguide-images/{filename}")
                log.info("Image uploaded: %s", public)
                return public
            else:
                log.error("Image upload failed: %d %s", r.status_code, r.text[:80])
                return None
        except Exception as e:
            log.error("Image upload error: %s", e)
            return None

    # ── Command handlers ──────────────────────────────────────────────────
    def handle_status(payload):
        log.info("CMD STATUS")
        s = last_state["scene_state"]
        f = last_state["fused"]
        d = last_state["detections"]
        if s is None:
            return
        cloud.log_alert(
            message="Remote status request",
            priority=1,
            zone_states=s.zones,
            closest_class=s.closest_class,
            closest_region=s.closest_region,
            closest_proximity=s.closest_proximity,
        )
        cloud.log_sensor(
            sensor_cm=f.sensor_cm if f else None,
            sensor_band=f.proximity if f else "none",
            object_count=len(d),
            confirmed_count=sum(1 for x in d if x["confirmed"]),
        )

    def handle_set_volume(payload):
        vol = int(payload.get("volume", 100))
        speech.set_volume(vol)
        log.info("CMD SET_VOLUME %d", vol)

    def handle_night_mode(payload):
        nonlocal night_mode_enabled
        night_mode_enabled = bool(payload.get("enable", True))
        log.info("CMD NIGHT_MODE %s", night_mode_enabled)

    def handle_request_image(payload):
        log.info("CMD REQUEST_IMAGE")
        frame = last_state.get("frame")
        if frame is None:
            log.warning("No frame available for capture")
            return
        public_url = upload_image(frame)
        if public_url:
            s = last_state["scene_state"]
            cloud.log_alert(
                message=f"Image captured: {public_url}",
                priority=1,
                zone_states=s.zones if s else {},
                closest_class="",
                closest_region="",
                closest_proximity="none",
            )

    cloud.register_command_handler("STATUS",        handle_status)
    cloud.register_command_handler("SET_VOLUME",    handle_set_volume)
    cloud.register_command_handler("NIGHT_MODE",    handle_night_mode)
    cloud.register_command_handler("REQUEST_IMAGE", handle_request_image)

    log.info("Inference ready")

    frames_done = 0
    fps_clock   = time.time()
    sensor_tick = 0

    while not stop.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # Night mode brightness boost
        if night_mode_enabled:
            frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)

        processed, detections = detector.get_detections(frame)
        scene_state           = scene.analyze(detections, frame.shape[1])
        dist_cm               = sensor.read_distance_cm()
        fused                 = fusion.fuse(dist_cm, scene_state)
        message, priority     = guidance.decide(
            scene_state, detections, speech, fused
        )

        # Update shared state for command handlers
        last_state["scene_state"] = scene_state
        last_state["detections"]  = detections
        last_state["fused"]       = fused
        last_state["frame"]       = processed

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
    from server import app   # imported inside thread
    app.run(host="0.0.0.0", port=port, threaded=True,
            debug=False, use_reloader=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VisionGuide launcher")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--port",   type=int, default=5000)
    args = parser.parse_args()

    log.info("Scanning for camera...")
    cam_index = find_camera(args.camera)
    if cam_index == -1:
        log.error("No working camera found — check USB connection")
        log.error("Try: python3 run.py --camera 1")
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

    # Start Flask first so browser can connect even before camera is ready
    flask_t = threading.Thread(
        target=server_loop, args=(args.port,),
        daemon=True, name="flask",
    )
    flask_t.start()
    time.sleep(1.2)
    log.info("Dashboard → http://%s:%d", ip, args.port)

    cap_t = threading.Thread(
        target=capture_loop,
        args=(cam_index, frame_q, stop),
        daemon=True, name="capture",
    )
    cap_t.start()

    inf_t = threading.Thread(
        target=inference_loop,
        args=(frame_q, stop),
        daemon=True, name="inference",
    )
    inf_t.start()

    try:
        while not stop.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("Stopping...")
        stop.set()

    inf_t.join(timeout=8)
    cap_t.join(timeout=3)
    log.info("Stopped.")


if __name__ == "__main__":
    main()