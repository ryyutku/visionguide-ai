# run.py  —  VisionGuide launcher with decoupled pipeline
#
# 3 threads running in parallel:
#
#   CAPTURE   — reads camera as fast as possible, keeps only latest frame
#               (drops stale frames so YOLO never processes old data)
#
#   INFERENCE — pulls latest frame, runs YOLO + guidance + sensor fusion
#               logs actual FPS every 5s so you can see performance
#
#   SERVER    — Flask serves MJPEG + JSON, completely independent of YOLO
#               the browser stream never stalls waiting for inference
#
# Usage:
#   python run.py                 # camera 0, port 5000
#   python run.py --camera 1      # if USB camera is /dev/video1
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


# ── Thread 1: Capture ─────────────────────────────────────────────────────────

def capture_loop(camera_index: int, frame_queue: queue.Queue,
                 stop: threading.Event):
    import cv2
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimum buffer = freshest frames

    if not cap.isOpened():
        log.error("Cannot open camera %d — try --camera 1", camera_index)
        stop.set()
        return

    log.info("Capture thread ready (camera %d)", camera_index)

    while not stop.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        # Keep only the latest frame — drop stale ones
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)

    cap.release()
    log.info("Capture thread stopped")


# ── Thread 2: Inference ───────────────────────────────────────────────────────

def inference_loop(frame_queue: queue.Queue, stop: threading.Event):
    from detector      import DetectorTracker
    from scene         import SceneAnalyzer
    from guidance      import GuidanceEngine
    from speech        import SpeechEngine, PRIORITY_HIGH
    from ultrasonic    import UltrasonicSensor
    from sensor_fusion import SensorFusion
    import shared_state

    detector = DetectorTracker()   # auto-picks NCNN if available
    scene    = SceneAnalyzer()
    guidance = GuidanceEngine()
    speech   = SpeechEngine()
    sensor   = UltrasonicSensor()
    fusion   = SensorFusion()

    log.info("Inference thread ready")

    frames_done = 0
    fps_clock   = time.time()

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

        if message:
            log.info("[ALERT p%d] %s", priority, message)
            if priority < PRIORITY_HIGH:
                speech.say(message, priority)

        shared_state.update_frame(processed)
        shared_state.update_state(
            scene_state, detections, message, priority, fused
        )

        # Report real FPS every 5 seconds
        frames_done += 1
        elapsed = time.time() - fps_clock
        if elapsed >= 5.0:
            log.info("Inference %.1f FPS", frames_done / elapsed)
            frames_done = 0
            fps_clock   = time.time()

    speech.shutdown()
    sensor.close()
    log.info("Inference thread stopped")


# ── Thread 3: Flask server ────────────────────────────────────────────────────

def server_loop(port: int):
    from server import app
    app.run(host="0.0.0.0", port=port, threaded=True,
            debug=False, use_reloader=False)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VisionGuide launcher")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port",   type=int, default=5000)
    args = parser.parse_args()

    ip = get_ip()
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║  VisionGuide                                 ║")
    print(f"  ║  Dashboard → http://{ip}:{args.port}".ljust(49) + "║")
    print(f"  ║  Camera    → index {args.camera}".ljust(49) + "║")
    print("  ║  Ctrl+C to stop                              ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()

    stop    = threading.Event()
    frame_q = queue.Queue(maxsize=1)

    # Start Flask first — browser can connect before camera is ready
    flask_t = threading.Thread(
        target=server_loop, args=(args.port,),
        daemon=True, name="flask",
    )
    flask_t.start()
    time.sleep(1.2)
    log.info("Dashboard → http://%s:%d", ip, args.port)

    # Start capture
    cap_t = threading.Thread(
        target=capture_loop, args=(args.camera, frame_q, stop),
        daemon=True, name="capture",
    )
    cap_t.start()

    # Start inference
    inf_t = threading.Thread(
        target=inference_loop, args=(frame_q, stop),
        daemon=True, name="inference",
    )
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