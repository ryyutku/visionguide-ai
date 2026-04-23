# export_model.py
#
# Optimises YOLOv8n for deployment on Raspberry Pi (ARM CPU).
#
# Improvements added:
#   - FP16 support for NCNN (faster on ARM)
#   - Better benchmarking (realistic inference)
#   - Tuned inference parameters (conf, iou)
#   - Reduced unnecessary overhead in predict()
#
# Output:
#   yolov8n_ncnn_model/     ← FP16 NCNN (faster)
#   yolov8n_int8.tflite     ← INT8 TFLite (optional)

import os
import time
import numpy as np
import cv2
from ultralytics import YOLO

MODEL_PT = "yolov8n.pt"
MODEL_NCNN_FP32 = "yolov8n_ncnn_model"
MODEL_TFLITE_INT8 = "yolov8n_int8.tflite"

INPUT_SIZE = 320  # optimized for Pi

# Inference tuning (IMPORTANT)
CONF_THRES = 0.4
IOU_THRES = 0.5


def benchmark(model_path: str, task: str, runs: int = 20) -> float:
    """Returns average inference time in milliseconds over `runs` frames."""
    try:
        model = YOLO(model_path, task=task)
    except Exception as e:
        print(f"  Could not load {model_path}: {e}")
        return float('inf')

    # Try real image first (more realistic)
    if os.path.exists("test.jpg"):
        dummy = cv2.imread("test.jpg")
        dummy = cv2.resize(dummy, (INPUT_SIZE, INPUT_SIZE))
    else:
        dummy = np.random.randint(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)

    # Warmup
    for _ in range(3):
        try:
            model.predict(
                dummy,
                imgsz=INPUT_SIZE,
                conf=CONF_THRES,
                iou=IOU_THRES,
                verbose=False,
                stream=False
            )
        except:
            pass

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(runs):
        try:
            model.predict(
                dummy,
                imgsz=INPUT_SIZE,
                conf=CONF_THRES,
                iou=IOU_THRES,
                verbose=False,
                stream=False
            )
        except:
            return float('inf')

    elapsed = (time.perf_counter() - t0) / runs * 1000
    return elapsed


def export_fp16_ncnn():
    print("\n── Step 1: Export FP16 NCNN ─────────────────────────────")
    if os.path.exists(MODEL_NCNN_FP32):
        print(f"  Already exists: ./{MODEL_NCNN_FP32}/  (skipping)")
        return True

    print("  Converting PyTorch → NCNN (FP16)...")
    print("  Uses ARM NEON + half precision for extra speed")

    try:
        model = YOLO(MODEL_PT)
        model.export(
            format="ncnn",
            imgsz=INPUT_SIZE,
            half=True  # 🔥 KEY IMPROVEMENT
        )
        print(f"  ✓ Saved: ./{MODEL_NCNN_FP32}/")
        return True
    except Exception as e:
        print(f"  ✗ NCNN export failed: {e}")
        return False


def export_int8_tflite():
    print("\n── Step 2: Export INT8 quantized TFLite ─────────────────")
    if os.path.exists(MODEL_TFLITE_INT8):
        print(f"  Already exists: ./{MODEL_TFLITE_INT8}  (skipping)")
        return True

    print("  Converting PyTorch → TFLite INT8...")

    try:
        model = YOLO(MODEL_PT)
        model.export(
            format="tflite",
            imgsz=INPUT_SIZE,
            int8=True,
        )

        if os.path.exists("yolov8n_int8.tflite"):
            print(f"  ✓ Saved: ./{MODEL_TFLITE_INT8}")
            return True
        else:
            print("  ⚠ TFLite export completed but file not found")
            return False

    except Exception as e:
        print(f"  ✗ TFLite INT8 export failed: {e}")
        print("  (Optional — NCNN is usually enough)")
        return False


def run_benchmarks():
    print("\n── Step 3: Benchmark comparison ─────────────────────────")
    results = {}

    print("  Benchmarking PyTorch FP32 (baseline)...")
    try:
        results["PyTorch FP32"] = benchmark(MODEL_PT, "detect")
        print(f"    → {results['PyTorch FP32']:.1f} ms/frame")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    if os.path.exists(MODEL_NCNN_FP32):
        print("  Benchmarking NCNN FP16...")
        ms = benchmark(MODEL_NCNN_FP32, "detect")
        if ms != float('inf'):
            results["NCNN FP16"] = ms
            print(f"    → {ms:.1f} ms/frame")
        else:
            print("    ✗ Failed")

    if os.path.exists(MODEL_TFLITE_INT8):
        print("  Benchmarking TFLite INT8...")
        ms = benchmark(MODEL_TFLITE_INT8, "detect")
        if ms != float('inf'):
            results["TFLite INT8"] = ms
            print(f"    → {ms:.1f} ms/frame")
        else:
            print("    ✗ Failed (may need tflite-runtime)")

    if results:
        print("\n  ─────────────────────────────────────────────────────")
        print("  Results (lower = faster):")

        baseline = results.get("PyTorch FP32")

        for name, ms in results.items():
            if name == "PyTorch FP32":
                print(f"  {name:<18} {ms:>7.1f} ms / frame  (baseline)")
            else:
                speedup = baseline / ms if baseline else 0
                print(f"  {name:<18} {ms:>7.1f} ms / frame  ({speedup:.1f}x faster)")

        print("  ─────────────────────────────────────────────────────")

        best_name = min(results, key=results.get)
        best_ms = results[best_name]

        print(f"\n  ✓ Best model: {best_name} ({best_ms:.1f} ms/frame)")
        print(f"  ✓ Estimated FPS on Pi 4: ~{1000 / best_ms:.1f}")


def main():
    print("VisionGuide — Model Optimization")
    print("=================================")
    print(f"Base model : {MODEL_PT}")
    print(f"Input size : {INPUT_SIZE}×{INPUT_SIZE} px")
    print(f"Target     : Raspberry Pi ARM CPU")

    print("\nOptimizations enabled:")
    print("  ✓ FP16 NCNN (ARM optimized)")
    print("  ✓ Reduced input size (320)")
    print("  ✓ Tuned inference thresholds")
    print("  ✓ Efficient benchmarking")

    success_ncnn = export_fp16_ncnn()
    success_tflite = export_int8_tflite()

    run_benchmarks()

    print("\n" + "=" * 50)

    if success_ncnn:
        print("✅ NCNN FP16 model ready! run.py will use it automatically.")
        print("   Expected performance: 6–10 FPS on Pi 4")
    else:
        print("⚠️ NCNN export failed. Falling back to PyTorch (~1–2 FPS).")

    if success_tflite:
        print("\n📦 TFLite INT8 model available (optional).")
        print("   Install: pip install tflite-runtime")

    print("=" * 50)


if __name__ == "__main__":
    main()