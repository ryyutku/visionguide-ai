# speech.py
#
# Windows : persistent PowerShell + System.Speech (non-blocking, interrupt support)
# Linux/Pi: espeak via subprocess (non-blocking, interrupt support)
#           Install with:  sudo apt install espeak
#
# Both backends support say_urgent() which cancels whatever is currently
# playing and speaks immediately — critical for a navigation aid.

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

        # Windows: persistent PowerShell process
        self._ps_proc = None
        if IS_WINDOWS:
            self._ps_proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden",
                 "-NonInteractive", "-Command", _PS_INIT],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log.info("Speech engine ready (PowerShell)")
        else:
            # Linux/Pi: check espeak is available
            result = subprocess.run(["which", "espeak"], capture_output=True)
            if result.returncode != 0:
                log.warning(
                    "espeak not found — run: sudo apt install espeak\n"
                    "Speech will be silent until installed."
                )
            else:
                log.info("Speech engine ready (espeak)")

        # Linux/Pi: track current espeak process so we can kill it for interrupts
        self._espeak_proc = None
        self._espeak_lock = threading.Lock()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ── Public API ────────────────────────────────────────────────────────

    def say(self, text: str, priority: int = PRIORITY_LOW):
        """Queue a message. Higher priority replaces a pending lower-priority one."""
        with self._lock:
            if self._pending_text is None or priority >= self._pending_pri:
                self._pending_text = text
                self._pending_pri  = priority
                self._event.set()

    def say_urgent(self, text: str):
        """Interrupt whatever is playing and speak immediately."""
        with self._lock:
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
        self._kill_espeak()

    # ── Worker thread ─────────────────────────────────────────────────────

    def _worker(self):
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

            interrupt = text.startswith("\x01")
            clean     = text[1:] if interrupt else text

            log.info("Speaking%s: '%s'", " [INTERRUPT]" if interrupt else "", clean)

            try:
                if IS_WINDOWS and self._ps_proc:
                    prefix = "!" if interrupt else ""
                    self._ps_proc.stdin.write((prefix + clean + "\n").encode("utf-8"))
                    self._ps_proc.stdin.flush()
                else:
                    self._speak_espeak(clean, interrupt)
            except Exception as exc:
                log.error("Speech error: %s", exc)
                if IS_WINDOWS:
                    self._restart_ps()

    def _speak_espeak(self, text: str, interrupt: bool):
        """
        Speak using espeak as a subprocess.
        If interrupt=True, kill any currently-running espeak first.
        espeak runs asynchronously so it doesn't block the worker thread —
        the worker can pick up the next queued message immediately.
        """
        if interrupt:
            self._kill_espeak()

        # espeak flags:
        #   -s 150  = speed (words per minute) — tune to taste
        #   -a 200  = amplitude (volume 0-200)
        #   --stdout piped to /dev/null so it runs non-blocking
        proc = subprocess.Popen(
            ["espeak", "-s", "150", "-a", "200", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with self._espeak_lock:
            self._espeak_proc = proc

    def _kill_espeak(self):
        with self._espeak_lock:
            if self._espeak_proc and self._espeak_proc.poll() is None:
                try:
                    self._espeak_proc.terminate()
                    self._espeak_proc.wait(timeout=0.5)
                except Exception:
                    pass
            self._espeak_proc = None

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
