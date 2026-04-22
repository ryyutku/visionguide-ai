# export_model.py
#
# Optimises YOLOv8n for deployment on Raspberry Pi (ARM CPU).
#
# Available optimizations for Pi:
#
#   1. FORMAT: PyTorch (.pt) → NCNN
#      NCNN is an inference framework built for ARM processors.
#      It uses ARM NEON SIMD instructions for ~2-3x speedup.
#
#   2. FORMAT: PyTorch (.pt) → TFLite INT8
#      TensorFlow Lite with INT8 quantization gives ~4x smaller model
#      and runs on CPU with integer acceleration.
#
#   3. INPUT SIZE: 640 → 320
#      Reducing resolution gives quadratic speedup with minimal accuracy loss.
#
# Usage:
#   python export_model.py
#
# Output:
#   yolov8n_ncnn_model/     ← FP32 NCNN (fast, works immediately)
#   yolov8n_int8.tflite     ← INT8 TFLite (optional, needs tflite-runtime)
#
# run.py will automatically use the fastest available model.

import os
import time
import numpy as np
from ultralytics import YOLO

MODEL_PT = "yolov8n.pt"
MODEL_NCNN_FP32 = "yolov8n_ncnn_model"
MODEL_TFLITE_FP32 = "yolov8n_float32.tflite"
MODEL_TFLITE_INT8 = "yolov8n_int8.tflite"

INPUT_SIZE = 320  # matches detector.py at runtime


def benchmark(model_path: str, task: str, runs: int = 20) -> float:
    """Returns average inference time in milliseconds over `runs` frames."""
    try:
        model = YOLO(model_path, task=task)
    except Exception as e:
        print(f"  Could not load {model_path}: {e}")
        return float('inf')

    dummy = np.random.randint(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    # Warmup
    for _ in range(3):
        try:
            model.predict(dummy, imgsz=INPUT_SIZE, verbose=False)
        except:
            pass
    # Timed runs
    t0 = time.perf_counter()
    for _ in range(runs):
        try:
            model.predict(dummy, imgsz=INPUT_SIZE, verbose=False)
        except:
            return float('inf')
    elapsed = (time.perf_counter() - t0) / runs * 1000
    return elapsed


def export_fp32_ncnn():
    print("\n── Step 1: Export FP32 NCNN ─────────────────────────────")
    if os.path.exists(MODEL_NCNN_FP32):
        print(f"  Already exists: ./{MODEL_NCNN_FP32}/  (skipping)")
        return True
    print("  Converting PyTorch → NCNN (FP32)...")
    print("  This uses ARM NEON optimizations for 2-3x speedup")
    try:
        model = YOLO(MODEL_PT)
        model.export(format="ncnn", imgsz=INPUT_SIZE)
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

    print("  Converting PyTorch → TFLite with INT8 quantization...")
    print("  This creates a 4x smaller model optimized for CPU")
    try:
        model = YOLO(MODEL_PT)
        model.export(
            format="tflite",
            imgsz=INPUT_SIZE,
            int8=True,
        )
        # Ultralytics exports to yolov8n_int8.tflite by default
        if os.path.exists(MODEL_TFLITE_INT8):
            print(f"  ✓ Saved: ./{MODEL_TFLITE_INT8}")
            return True
        elif os.path.exists("yolov8n_int8.tflite"):
            print(f"  ✓ Saved: ./{MODEL_TFLITE_INT8}")
            return True
        else:
            print("  ⚠ TFLite export completed but file not found")
            return False
    except Exception as e:
        print(f"  ✗ TFLite INT8 export failed: {e}")
        print("  (This is optional - NCNN already gives good performance)")
        return False


def run_benchmarks():
    print("\n── Step 3: Benchmark comparison ─────────────────────────")
    results = {}

    # PyTorch baseline
    print("  Benchmarking PyTorch FP32 (baseline)...")
    try:
        results["PyTorch FP32"] = benchmark(MODEL_PT, "detect")
        print(f"    → {results['PyTorch FP32']:.1f} ms/frame")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # NCNN
    if os.path.exists(MODEL_NCNN_FP32):
        print("  Benchmarking NCNN FP32...")
        ms = benchmark(MODEL_NCNN_FP32, "detect")
        if ms != float('inf'):
            results["NCNN FP32"] = ms
            print(f"    → {ms:.1f} ms/frame")
        else:
            print("    ✗ Failed")

    # TFLite INT8
    if os.path.exists(MODEL_TFLITE_INT8):
        print("  Benchmarking TFLite INT8...")
        ms = benchmark(MODEL_TFLITE_INT8, "detect")
        if ms != float('inf'):
            results["TFLite INT8"] = ms
            print(f"    → {ms:.1f} ms/frame")
        else:
            print("    ✗ Failed (may need tflite-runtime)")

    if results:
        print()
        print("  ─────────────────────────────────────────────────────")
        print("  Results (lower = faster):")
        baseline = results.get("PyTorch FP32")

        for name, ms in results.items():
            if name == "PyTorch FP32":
                print(f"  {name:<18} {ms:>7.1f} ms / frame  (baseline)")
            else:
                speedup = baseline / ms if baseline else 0
                print(f"  {name:<18} {ms:>7.1f} ms / frame  ({speedup:.1f}x faster)")
        print("  ─────────────────────────────────────────────────────")

        # Find best
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
    print()
    print("Note: INT8 quantization is not directly supported for NCNN.")
    print("      FP32 NCNN already gives 2-3x speedup using ARM NEON.")
    print("      TFLite INT8 is an alternative if you install tflite-runtime.")

    success_ncnn = export_fp32_ncnn()
    success_tflite = export_int8_tflite()
    run_benchmarks()

    print("\n" + "=" * 50)
    if success_ncnn:
        print("✅ NCNN model ready! run.py will use it automatically.")
        print("   Expected performance: 5-8 FPS on Pi 4")
    else:
        print("⚠️  NCNN export failed. run.py will fall back to PyTorch.")
        print("   Performance will be ~1-2 FPS.")

    if success_tflite:
        print("\n📦 TFLite INT8 model also available.")
        print("   To use it, install: pip install tflite-runtime")
        print("   Then modify detector.py to load TFLite models.")

    print("=" * 50)


if __name__ == "__main__":
    main()