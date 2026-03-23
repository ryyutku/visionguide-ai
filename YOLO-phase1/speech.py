# speech.py
# On Windows, uses PowerShell's built-in speech synthesizer directly.
# This bypasses pyttsx3 and its COM threading issues entirely.
# Falls back to pyttsx3 on non-Windows systems.

import threading
import subprocess
import sys
import logging

log = logging.getLogger("speech")

PRIORITY_HIGH   = 3
PRIORITY_MEDIUM = 2
PRIORITY_LOW    = 1

IS_WINDOWS = sys.platform == "win32"


def _speak_windows(text: str):
    """Speak using PowerShell SpeechSynthesizer — no COM setup needed."""
    safe = text.replace("'", "")   # strip single quotes to avoid PS injection
    cmd  = (
        f"Add-Type -AssemblyName System.Speech; "
        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Rate = 1; "
        f"$s.Speak('{safe}')"
    )
    subprocess.run(
        ["powershell", "-WindowStyle", "Hidden", "-Command", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _speak_pyttsx3(engine, text: str):
    engine.say(text)
    engine.runAndWait()


class SpeechEngine:
    def __init__(self):
        self._lock         = threading.Lock()
        self._pending_text = None
        self._pending_pri  = 0
        self._event        = threading.Event()
        self._running      = True
        self._thread       = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        log.info("Speech engine started (backend: %s)",
                 "PowerShell" if IS_WINDOWS else "pyttsx3")

    def say(self, text: str, priority: int = PRIORITY_LOW):
        with self._lock:
            if self._pending_text is None or priority >= self._pending_pri:
                self._pending_text = text
                self._pending_pri  = priority
                self._event.set()

    def shutdown(self):
        self._running = False
        self._event.set()
        self._thread.join(timeout=6)

    def _worker(self):
        engine = None
        if not IS_WINDOWS:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 165)
                engine.setProperty("volume", 1.0)
            except Exception as e:
                log.error("pyttsx3 init failed: %s", e)

        while self._running:
            self._event.wait()
            self._event.clear()

            if not self._running:
                break

            with self._lock:
                text               = self._pending_text
                self._pending_text = None
                self._pending_pri  = 0

            if not text:
                continue

            log.info("Speaking: '%s'", text)
            try:
                if IS_WINDOWS:
                    _speak_windows(text)
                else:
                    _speak_pyttsx3(engine, text)
            except Exception as exc:
                log.error("Speech error: %s", exc)