# speech.py
#
# Windows : persistent PowerShell + System.Speech
# Linux/Pi: persistent espeak process (kept alive — no spawn delay per alert)
#
# Both support say_urgent() which kills current audio and speaks immediately.
# Install espeak on Pi:  sudo apt install espeak


import threading
import subprocess
import sys
import logging

log = logging.getLogger("speech")

PRIORITY_HIGH   = 3
PRIORITY_MEDIUM = 2
PRIORITY_LOW    = 1

IS_WINDOWS = sys.platform == "win32"

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
        self._ps_proc      = None
        self._volume       = 100  # default volume (espeak -a parameter, range 0-200)

        if IS_WINDOWS:
            self._ps_proc = self._start_ps()
            log.info("Speech engine ready (PowerShell)")
        else:
            r = subprocess.run(["which", "espeak"], capture_output=True)
            if r.returncode != 0:
                log.warning("espeak not found — run: sudo apt install espeak")
            else:
                log.info("Speech engine ready (espeak)")

        self._espeak_proc = None
        self._espeak_lock = threading.Lock()

        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="speech-worker"
        )
        self._thread.start()

    def set_volume(self, volume: int):
        """Set speech volume (0-200)."""
        self._volume = max(0, min(200, volume))
        log.info("Volume set to %d", self._volume)

    def say(self, text: str, priority: int = PRIORITY_LOW):
        with self._lock:
            if self._pending_text is None or priority >= self._pending_pri:
                self._pending_text = text
                self._pending_pri  = priority
                self._event.set()

    def say_urgent(self, text: str):
        with self._lock:
            self._pending_text = "\x01" + text
            self._pending_pri  = PRIORITY_HIGH
            self._event.set()

    def shutdown(self):
        self._running = False
        self._event.set()
        self._thread.join(timeout=3)
        self._kill_espeak()
        if self._ps_proc:
            try:
                self._ps_proc.stdin.close()
                self._ps_proc.terminate()
            except Exception:
                pass

    def _worker(self):
        while self._running:
            self._event.wait()
            self._event.clear()
            if not self._running:
                break

            with self._lock:
                text = self._pending_text
                self._pending_text = None
                self._pending_pri  = 0

            if not text:
                continue

            interrupt = text.startswith("\x01")
            clean     = text[1:] if interrupt else text

            log.info("Speaking%s: '%s'", " [INTERRUPT]" if interrupt else "", clean)

            try:
                if IS_WINDOWS and self._ps_proc:
                    prefix = "!" if interrupt else ""
                    self._ps_proc.stdin.write((prefix + clean + "\n").encode("utf-8"))
                    self._ps_proc.stdin.flush()
                else:
                    self._speak_linux(clean, interrupt)
            except Exception as exc:
                log.error("Speech error: %s", exc)
                if IS_WINDOWS:
                    self._ps_proc = self._start_ps()

    def _speak_linux(self, text: str, interrupt: bool):
        if interrupt:
            self._kill_espeak()

        with self._espeak_lock:
            if not interrupt and self._espeak_proc is not None:
                if self._espeak_proc.poll() is None:
                    return

            # Use current volume setting
            proc = subprocess.Popen(
                ["espeak", "-s", "160", "-a", str(self._volume), "--", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._espeak_proc = proc

    def _kill_espeak(self):
        with self._espeak_lock:
            if self._espeak_proc and self._espeak_proc.poll() is None:
                try:
                    self._espeak_proc.terminate()
                    self._espeak_proc.wait(timeout=0.3)
                except Exception:
                    pass
            self._espeak_proc = None

    def _start_ps(self):
        return subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden",
             "-NonInteractive", "-Command", _PS_INIT],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )