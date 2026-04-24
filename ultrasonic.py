# ultrasonic.py
#
# Auto-selects backend:
#   Raspberry Pi → GPIO (RPi.GPIO)
#   Laptop/other → Stub (simulated oscillating distance)
#
# Wiring (Pi, BCM numbering):
#   HC-SR04 VCC  → Pi pin 2  (5V)
#   HC-SR04 GND  → Pi pin 6  (GND)
#   HC-SR04 TRIG → GPIO 23   (override: ULTRASONIC_TRIG=XX)
#   HC-SR04 ECHO → GPIO 24   (override: ULTRASONIC_ECHO=XX)
#
#   IMPORTANT: voltage divider on ECHO (sensor outputs 5V, Pi GPIO = 3.3V max):
#       ECHO → 1kΩ → GPIO 24
#                 ↘ 2kΩ → GND
#
# Threshold rationale (latency compensation):
#   Pipeline latency on Pi 4 ≈ 250–350ms
#   (sensor poll 60ms + YOLO 160ms + speech startup 50ms + margin)
#   At walking pace ~1.2 m/s, the user travels ~36cm during processing.
#   So trigger distances are set ~35cm higher than the desired real-world
#   warning distance to ensure the alert arrives on time.
#
#   DIST_CRITICAL = 75cm  → alert arrives when object is ~40cm away
#   DIST_CLOSE    = 110cm → alert arrives when object is ~75cm away
#
#   MUST stay consistent with guidance.py:
#     STOP_DISTANCE_CM = 75  (matches DIST_CRITICAL)
#     CLEAR_SENSOR_CM  = 120 (must be > DIST_CLOSE so clear only fires when path is open)

import os
import time
import threading
import logging

log = logging.getLogger("ultrasonic")

# ── Distance thresholds (latency-compensated) ─────────────────────────────────
DIST_CRITICAL = 75    # cm — sensor "critical" band → triggers hard stop
DIST_CLOSE    = 110   # cm — sensor "close" band
DIST_MEDIUM   = 160   # cm — sensor "medium" band
DIST_MAX      = 400   # cm — beyond this is noise, discard


def _detect_backend() -> str:
    explicit = os.environ.get("ULTRASONIC_BACKEND", "").lower()
    if explicit in ("gpio", "stub"):
        return explicit
    try:
        with open("/proc/device-tree/model") as f:
            if "raspberry" in f.read().lower():
                return "gpio"
    except (FileNotFoundError, PermissionError):
        pass
    return "stub"


class _SensorBase:
    def read_distance_cm(self) -> float | None:
        raise NotImplementedError

    def proximity_band(self, cm: float | None) -> str:
        if cm is None:         return "none"
        if cm < DIST_CRITICAL: return "critical"
        if cm < DIST_CLOSE:    return "close"
        if cm < DIST_MEDIUM:   return "medium"
        return "far"

    def close(self):
        pass


class _GPIOSensor(_SensorBase):
    TRIG = int(os.environ.get("ULTRASONIC_TRIG", "23"))
    ECHO = int(os.environ.get("ULTRASONIC_ECHO", "24"))

    def __init__(self):
        import RPi.GPIO as GPIO
        self._gpio = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.TRIG, GPIO.OUT)
        GPIO.setup(self.ECHO, GPIO.IN)
        GPIO.output(self.TRIG, False)
        time.sleep(0.05)

        self._latest: float | None = None
        self._lock    = threading.Lock()
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="ultrasonic"
        )
        self._thread.start()
        log.info("GPIO ultrasonic ready  TRIG=%d  ECHO=%d", self.TRIG, self.ECHO)

    def _loop(self):
        while self._running:
            cm = self._ping()
            with self._lock:
                self._latest = cm
            time.sleep(0.06)   # ~16 Hz

    def _ping(self) -> float | None:
        GPIO = self._gpio
        GPIO.output(self.TRIG, True)
        time.sleep(0.00001)
        GPIO.output(self.TRIG, False)

        t = time.time()
        while GPIO.input(self.ECHO) == 0:
            if time.time() - t > 0.02:
                return None
        t0 = time.time()

        while GPIO.input(self.ECHO) == 1:
            if time.time() - t0 > 0.04:
                return None
        t1 = time.time()

        cm = (t1 - t0) * 17150
        return cm if cm < DIST_MAX else None

    def read_distance_cm(self) -> float | None:
        with self._lock:
            return self._latest

    def close(self):
        self._running = False
        self._thread.join(timeout=1)
        self._gpio.cleanup()


class _StubSensor(_SensorBase):
    def __init__(self):
        self._start = time.time()
        self._fixed = None
        v = os.environ.get("ULTRASONIC_STUB_FIXED")
        if v:
            try:
                self._fixed = float(v)
            except ValueError:
                pass
        mode = f"fixed {self._fixed}cm" if self._fixed else "oscillating 30–200cm"
        log.info("Stub ultrasonic active (%s)", mode)

    def read_distance_cm(self) -> float | None:
        if self._fixed is not None:
            return self._fixed
        import math
        t = time.time() - self._start
        return 115 + 85 * math.sin(t * 2 * math.pi / 20)


class UltrasonicSensor(_SensorBase):
    def __init__(self):
        backend = _detect_backend()
        log.info("Ultrasonic backend: %s", backend)
        self._impl = _GPIOSensor() if backend == "gpio" else _StubSensor()

    def read_distance_cm(self) -> float | None:
        return self._impl.read_distance_cm()

    def proximity_band(self, cm: float | None) -> str:
        return self._impl.proximity_band(cm)

    def close(self):
        self._impl.close()