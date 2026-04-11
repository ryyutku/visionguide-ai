# export_model.py
#
# Run this ONCE on the Raspberry Pi to convert the model to NCNN format.
# NCNN is optimised for ARM processors — expect 2-3x faster inference.
#
# Usage:
#   python export_model.py
#
# Output: creates a folder called yolov8n_ncnn_model/ in the same directory.
# After running this, run.py will automatically use the NCNN model.

from ultralytics import YOLO
import os

MODEL_PT   = "yolov8n.pt"
MODEL_NCNN = "yolov8n_ncnn_model"

if os.path.exists(MODEL_NCNN):
    print(f"NCNN model already exists at ./{MODEL_NCNN}/")
    print("Delete the folder and re-run if you want to re-export.")
else:
    print("Exporting YOLOv8n to NCNN format...")
    print("This takes 1-2 minutes on Pi — only needs to be done once.")
    model = YOLO(MODEL_PT)
    model.export(format="ncnn", imgsz=320)
    print(f"\nDone. NCNN model saved to ./{MODEL_NCNN}/")
    print("Run 'python run.py' to start with the optimised model.")