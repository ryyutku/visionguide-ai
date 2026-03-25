# voice_input.py
# Listens for voice commands in a background thread.
# Commands:
#   "find <object>"   — set new search target
#   "got it"          — user has picked up / found the object
#   "cancel"          — stop searching
#
# Requires: pip install SpeechRecognition pyaudio
# On Windows pyaudio install: pip install pipwin && pipwin install pyaudio

import threading
import logging

log = logging.getLogger("voice_input")

# Try to import speech recognition — fail gracefully if not installed
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    log.warning("SpeechRecognition not installed — voice commands disabled. "
                "Install with: pip install SpeechRecognition pyaudio")


class VoiceInput:
    """
    Background listener. Call .get_command() each frame to check
    if a command has arrived. Returns None if nothing new.
    """

    def __init__(self):
        self._command  = None
        self._lock     = threading.Lock()
        self._running  = False
        self._thread   = None

        if SR_AVAILABLE:
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold        = 300
            self._recognizer.dynamic_energy_threshold = True
            self._running = True
            self._thread  = threading.Thread(
                target=self._listen_loop, daemon=True)
            self._thread.start()
            log.info("Voice input listening")
        else:
            log.info("Voice input disabled (SpeechRecognition not available)")

    def get_command(self) -> dict | None:
        """
        Returns a command dict or None.
        Command dict: {"action": "find"|"got_it"|"cancel", "target": str|None}
        """
        with self._lock:
            cmd = self._command
            self._command = None
            return cmd

    def shutdown(self):
        self._running = False

    # ------------------------------------------------------------------

    def _listen_loop(self):
        with sr.Microphone() as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=1)
            log.info("Microphone ready")

            while self._running:
                try:
                    audio = self._recognizer.listen(
                        source, timeout=3, phrase_time_limit=5)
                    text  = self._recognizer.recognize_google(audio).lower()
                    log.info("Heard: '%s'", text)
                    cmd   = self._parse(text)
                    if cmd:
                        with self._lock:
                            self._command = cmd
                        log.info("Command: %s", cmd)
                except sr.WaitTimeoutError:
                    pass   # silence — keep listening
                except sr.UnknownValueError:
                    pass   # couldn't understand
                except Exception as e:
                    log.error("Voice input error: %s", e)

    def _parse(self, text: str) -> dict | None:
        # "find <object>" or "look for <object>" or "search for <object>"
        for prefix in ("find ", "look for ", "search for ", "where is ", "where's "):
            if text.startswith(prefix):
                target = text[len(prefix):].strip()
                if target:
                    return {"action": "find", "target": target}

        # "got it" / "i got it" / "found it" / "i have it" / "picked it up"
        if any(p in text for p in ("got it", "found it",
                                    "i have it", "picked it", "i got it")):
            return {"action": "got_it", "target": None}

        # "cancel" / "stop" / "never mind"
        if any(p in text for p in ("cancel", "stop searching", "never mind")):
            return {"action": "cancel", "target": None}

        return None