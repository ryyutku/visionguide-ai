# speech.py
# Persistent PowerShell process with interrupt support.
# High-priority messages (center alerts) cancel whatever is currently speaking
# and start immediately — so a side warning never blocks a center alert.

import threading
import subprocess
import sys
import logging

log = logging.getLogger("speech")

PRIORITY_HIGH   = 3
PRIORITY_MEDIUM = 2
PRIORITY_LOW    = 1

IS_WINDOWS = sys.platform == "win32"

# Protocol: lines prefixed with "!" = cancel current speech then speak immediately.
#           lines with no prefix   = speak normally (won't interrupt current audio).
_PS_INIT = (
    "Add-Type -AssemblyName System.Speech; "
    "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$synth.Rate = 2; "
    "$synth.Volume = 100; "
    "while ($true) { "
    "  $line = [Console]::ReadLine(); "
    "  if ($line -eq $null) { break } "
    "  if ($line -eq '') { continue } "
    "  if ($line.StartsWith('!')) { "
    "    $synth.SpeakAsyncCancelAll(); "
    "    $synth.SpeakAsync($line.Substring(1)) | Out-Null "
    "  } else { "
    "    $synth.SpeakAsync($line) | Out-Null "
    "  } "
    "}"
)


class SpeechEngine:
    def __init__(self):
        self._lock         = threading.Lock()
        self._pending_text = None
        self._pending_pri  = 0
        self._event        = threading.Event()
        self._running      = True

        self._ps_proc = None
        if IS_WINDOWS:
            self._ps_proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden",
                 "-NonInteractive", "-Command", _PS_INIT],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log.info("Speech engine ready (persistent PowerShell)")
        else:
            log.info("Speech engine ready (pyttsx3)")

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def say(self, text: str, priority: int = PRIORITY_LOW):
        """Queue a message. Higher priority replaces lower priority pending messages."""
        with self._lock:
            if self._pending_text is None or priority >= self._pending_pri:
                self._pending_text = text
                self._pending_pri  = priority
                self._event.set()

    def say_urgent(self, text: str):
        """
        Speak immediately, cancelling whatever is currently playing.
        Use this for center/high-priority alerts that must not be delayed.
        """
        with self._lock:
            # Override everything — mark with INTERRUPT flag
            self._pending_text = "\x01" + text   # \x01 = interrupt marker
            self._pending_pri  = PRIORITY_HIGH
            self._event.set()

    def shutdown(self):
        self._running = False
        self._event.set()
        self._thread.join(timeout=4)
        if self._ps_proc:
            try:
                self._ps_proc.stdin.close()
                self._ps_proc.terminate()
            except Exception:
                pass

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

            # Check for interrupt marker
            interrupt = text.startswith("\x01")
            clean     = text[1:] if interrupt else text

            log.info("Speaking%s: '%s'", " [INTERRUPT]" if interrupt else "", clean)

            try:
                if IS_WINDOWS and self._ps_proc:
                    # "!" prefix tells PS to cancel current speech first
                    prefix = "!" if interrupt else ""
                    self._ps_proc.stdin.write((prefix + clean + "\n").encode("utf-8"))
                    self._ps_proc.stdin.flush()
                elif engine:
                    engine.say(clean)
                    engine.runAndWait()
            except Exception as exc:
                log.error("Speech error: %s", exc)
                if IS_WINDOWS:
                    self._restart_ps()

    def _restart_ps(self):
        log.info("Restarting PowerShell speech process...")
        try:
            if self._ps_proc:
                self._ps_proc.terminate()
        except Exception:
            pass
        try:
            self._ps_proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden",
                 "-NonInteractive", "-Command", _PS_INIT],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            log.error("Failed to restart PS process: %s", e)