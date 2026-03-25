# search_main.py
#
# Controls:
#   Voice command  — "find cup", "find chair", "got it", "cancel"
#   Keyboard (debug only):
#     G  — "got it" (picked up)
#     Q  — quit

import cv2
import sys
import logging
import time

for _n in ["comtypes", "comtypes.client", "comtypes.server",
           "PIL", "ultralytics", "torch", "urllib3",
           "pyttsx3", "pyttsx3.driver"]:
    logging.getLogger(_n).setLevel(logging.CRITICAL)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-10s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

from search_detector import SearchDetector
from search_guidance import SearchGuidance, PICKED_UP, IDLE
from speech          import SpeechEngine
from voice_input     import VoiceInput

# ── Supported objects (YOLO v8 indoor-relevant classes) ──────────────────────
SUPPORTED = {
    "chair", "couch", "bed", "dining table", "toilet",
    "cup", "bottle", "bowl", "laptop", "keyboard", "mouse",
    "remote", "cell phone", "book", "clock", "vase",
    "backpack", "handbag", "suitcase", "umbrella", "person",
}

DEFAULT_TARGET = sys.argv[1] if len(sys.argv) > 1 else "cup"


def main():
    detector = SearchDetector("yolov8n.pt")   # fast model
    guidance = SearchGuidance(DEFAULT_TARGET)
    speech   = SpeechEngine()
    voice    = VoiceInput()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    target  = DEFAULT_TARGET
    state   = "scanning"
    message = ""

    # Announce start
    speech.say(f"Search mode active. Say find and the object name to start. "
               f"Currently searching for {target}.")
    log.info("Searching for: %s", target)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Check voice commands ─────────────────────────────────────────
        cmd = voice.get_command()
        if cmd:
            if cmd["action"] == "find" and cmd["target"]:
                new_target = cmd["target"]
                log.info("Voice: find '%s'", new_target)
                if new_target not in SUPPORTED:
                    speech.say(f"I may not recognise {new_target}, but I will try.")
                target  = new_target
                message = ""
                guidance.reset(new_target)
                speech.say(f"Looking for {new_target}.")

            elif cmd["action"] == "got_it":
                log.info("Voice: got it")
                state, message = guidance.handle_got_it(speech)

            elif cmd["action"] == "cancel":
                log.info("Voice: cancel")
                speech.say("Search cancelled.")
                guidance.reset(target)

        # ── Process frame ────────────────────────────────────────────────
        if state != PICKED_UP:
            processed_frame, detections = detector.get_detections(frame, target)
            state, message              = guidance.process(detections, speech)
        else:
            processed_frame = cv2.flip(cv2.resize(frame, (640, 480)), 1)

        # ── Draw HUD ─────────────────────────────────────────────────────
        _draw_hud(processed_frame, target, state, message)
        cv2.imshow("VisionGuide — Search Mode", processed_frame)

        # ── Keyboard fallback (useful for testing without microphone) ────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            state, message = guidance.handle_got_it(speech)
        elif key == ord('r'):
            guidance.reset(target)
            speech.say(f"Restarting search for {target}.")
            log.info("Reset search")

    speech.shutdown()
    voice.shutdown()
    cap.release()
    cv2.destroyAllWindows()


def _draw_hud(frame, target: str, state: str, message: str):
    h, w = frame.shape[:2]

    state_colors = {
        "scanning":  (100, 100, 100),
        "guiding":   (0,   180, 255),
        "close":     (0,   220,  80),
        "found":     (0,   255,   0),
        "picked_up": (0,   255, 180),
        "idle":      (60,  60,  60),
    }
    color = state_colors.get(state, (180, 180, 180))

    cv2.putText(frame, f"Target: {target}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, state.upper(), (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Controls reminder bottom-right
    cv2.putText(frame, "G=got it  R=reset  Q=quit",
                (w - 210, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)

    if message:
        cv2.rectangle(frame, (0, h - 38), (w, h), (10, 10, 10), -1)
        cv2.putText(frame, message, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()