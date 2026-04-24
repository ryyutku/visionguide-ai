# Raspberry Pi Setup Guide — VisionGuide
# =========================================

# ── 1. HARDWARE YOU NEED ─────────────────────────────────────────────────────
#
#   Raspberry Pi 4 (2GB+ RAM recommended — YOLO needs ~1GB)
#   USB camera (any UVC-compatible webcam)
#   HC-SR04 ultrasonic sensor
#   Resistors: one 1kΩ and one 2kΩ (for ECHO voltage divider)
#   Jumper wires, breadboard
#   Speaker or headphones (3.5mm jack or USB)
#   Power bank (if making it portable)


# ── 2. SENSOR WIRING ─────────────────────────────────────────────────────────
#
#   HC-SR04 pin → Pi pin
#   VCC         → Pin 2   (5V)
#   GND         → Pin 6   (GND)
#   TRIG        → Pin 16  (GPIO 23, BCM)
#   ECHO        → [voltage divider] → Pin 18 (GPIO 24, BCM)
#
#   Voltage divider on ECHO (REQUIRED — sensor outputs 5V, Pi GPIO max is 3.3V):
#
#       HC-SR04 ECHO
#            │
#           1kΩ
#            │
#            ├──────────── GPIO 24 (Pi pin 18)
#            │
#           2kΩ
#            │
#           GND


# ── 3. RASPBERRY PI OS SETUP ─────────────────────────────────────────────────
#
# Start with Raspberry Pi OS Lite (64-bit) or Desktop (64-bit).
# Run these commands after first boot:

sudo apt update && sudo apt upgrade -y

# System dependencies
sudo apt install -y \
    python3-pip \
    python3-venv \
    espeak \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libjasper-dev \
    libhdf5-dev \
    git


# ── 4. PROJECT SETUP ─────────────────────────────────────────────────────────

# Create a folder for the project
mkdir ~/visionguide
cd ~/visionguide

# Copy all your .py files here:
#   detector.py  scene.py  speech.py  guidance.py
#   ultrasonic.py  sensor_fusion.py  main.py  ui.py

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install \
    ultralytics \
    opencv-python-headless \
    RPi.GPIO \
    pillow

# Download the YOLO model (happens automatically on first run,
# but you can pre-download it):
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"


# ── 5. RUNNING THE PROJECT ───────────────────────────────────────────────────

cd ~/visionguide
source venv/bin/activate

# Basic run (USB camera on /dev/video0):
python3 main.py

# If your USB camera is on a different index:
CAMERA_INDEX=1 python3 main.py

# Check which video devices are available:
ls /dev/video*

# Run with the UI dashboard instead:
python3 ui.py

# Force stub sensor for testing (ignores GPIO):
ULTRASONIC_BACKEND=stub python3 main.py

# Pin stub sensor to a fixed distance for testing guidance logic:
ULTRASONIC_BACKEND=stub ULTRASONIC_STUB_FIXED=35 python3 main.py


# ── 6. PERFORMANCE TIPS FOR PI ───────────────────────────────────────────────
#
# YOLOv8n on Pi 4 runs at ~3-6 FPS depending on load. This is acceptable
# for a walking-pace navigation aid. To squeeze out more speed:

# Option A: Export model to NCNN format (fastest on Pi, ~2x speedup)
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='ncnn')
"
# Then change model_path in main.py to 'yolov8n_ncnn_model'

# Option B: Reduce input resolution in detector.py
# Change: frame = cv2.resize(frame, (640, 480))
# To:     frame = cv2.resize(frame, (320, 240))
# Faster but less accurate at range.

# Option C: Increase conf threshold in detector.py to reduce detections
# Change: conf=0.4  →  conf=0.5


# ── 7. RUN ON BOOT (optional) ────────────────────────────────────────────────
#
# To start VisionGuide automatically when Pi powers on,
# create a systemd service:

sudo nano /etc/systemd/system/visionguide.service

# Paste this into the file:
# ---
# [Unit]
# Description=VisionGuide Navigation Aid
# After=multi-user.target
#
# [Service]
# Type=simple
# User=pi
# WorkingDirectory=/home/pi/visionguide
# ExecStart=/home/pi/visionguide/venv/bin/python3 main.py
# Restart=on-failure
# RestartSec=5
# Environment=DISPLAY=:0
#
# [Install]
# WantedBy=multi-user.target
# ---

sudo systemctl daemon-reload
sudo systemctl enable visionguide
sudo systemctl start visionguide

# Check status:
sudo systemctl status visionguide

# View logs:
journalctl -u visionguide -f


# ── 8. TROUBLESHOOTING ───────────────────────────────────────────────────────
#
# "Could not open camera index 0"
#   → ls /dev/video*   to find your camera
#   → Try CAMERA_INDEX=1 python3 main.py
#
# "espeak not found"
#   → sudo apt install espeak
#
# "RPi.GPIO not found"
#   → pip install RPi.GPIO
#   → Make sure you're running as a user in the gpio group:
#     sudo usermod -aG gpio pi
#     (then log out and back in)
#
# "ImportError: libopenblas"
#   → sudo apt install libatlas-base-dev
#
# YOLO runs very slowly
#   → Export to NCNN (see Performance Tips above)
#   → Make sure you're using the 64-bit Pi OS
#   → Check you're not thermally throttling: vcgencmd measure_temp
#
# Sensor always reads None
#   → Check wiring — especially the voltage divider on ECHO
#   → Double-check BCM pin numbers with: pinout   (run in terminal)
#   → Test sensor alone: ULTRASONIC_BACKEND=gpio python3 -c "
#        from ultrasonic import UltrasonicSensor
#        import time
#        s = UltrasonicSensor()
#        for _ in range(10):
#            print(s.read_distance_cm())
#            time.sleep(0.5)
#     "
