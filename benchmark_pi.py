import cv2
import time
import torch
import pandas as pd
from ultralytics import YOLO

# --- SIMULATION SETTINGS ---
# Raspberry Pi 4 has 4 cores, but they are weak. 
# Setting this to 1 or 2 on a PC usually mimics Pi 4 speed best.
torch.set_num_threads(1) 

MODELS_TO_TEST = ["yolov8n.pt", "yolov8s.pt"]
IMG_SIZE = 416  # As you chose, 416 is a great balance for Pi
NUM_FRAMES = 50 # How many frames to test per model

def run_benchmark(model_name):
    print(f"\n--- Starting Test for {model_name} ---")
    model = YOLO(model_name)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    # Warm-up phase (prevents the first slow frame from ruining the average)
    for _ in range(5):
        ret, frame = cap.read()
        _ = model.predict(frame, imgsz=IMG_SIZE, device='cpu', verbose=False)

    frame_times = []
    
    for i in range(NUM_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        # Inference
        results = model.predict(frame, imgsz=IMG_SIZE, device='cpu', verbose=False)
        end_time = time.time()

        frame_times.append(end_time - start_time)

        # Show the preview
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f"Model: {model_name} | Frame: {i}/{NUM_FRAMES}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("YOLO Pi Simulation Benchmark", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    avg_inference = sum(frame_times) / len(frame_times)
    fps = 1 / avg_inference
    return {"Model": model_name, "Avg Latency (ms)": f"{avg_inference*1000:.2f}", "Estimated FPS": f"{fps:.2f}"}

# Execute Tests
results_list = []
for m in MODELS_TO_TEST:
    res = run_benchmark(m)
    if res:
        results_list.append(res)

# Final Comparison Table
print("\n" + "="*40)
print("       BENCHMARK RESULTS")
print("="*40)
df = pd.DataFrame(results_list)
print(df.to_string(index=False))
print("="*40)
print("Note: On a real Pi 4, expect performance to be roughly 30-50% slower than this 'single-thread' PC test.")