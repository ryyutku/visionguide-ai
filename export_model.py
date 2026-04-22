# export_model.py
#
# Optimises YOLOv8n for deployment on Raspberry Pi (ARM CPU).
#
# Two optimizations are applied:
#
#   1. FORMAT: PyTorch (.pt) → NCNN
#      NCNN is an inference framework built for ARM processors.
#      It uses ARM NEON SIMD instructions to process multiple values
#      simultaneously, which PyTorch's general runtime cannot do.
#
#   2. QUANTIZATION: FP32 → INT8
#      Converts model weights from 32-bit floats to 8-bit integers.
#      - Model file size shrinks ~4x
#      - Memory bandwidth drops ~4x (Pi RAM bandwidth is a bottleneck)
#      - Integer arithmetic is faster than float on ARM Cortex-A
#      - Accuracy loss is typically < 1% mAP for detection tasks
#
# Usage:
#   python export_model.py
#
# Output:
#   yolov8n_ncnn_model/   ← FP32 NCNN (fallback)
#   yolov8n_int8_ncnn/    ← INT8 quantized NCNN (preferred, faster)

import os
import time
import numpy as np
from ultralytics import YOLO

MODEL_PT        = "yolov8n.pt"
MODEL_NCNN_FP32 = "yolov8n_ncnn_model"
MODEL_NCNN_INT8 = "yolov8n_int8_ncnn"

INPUT_SIZE = 320   # matches detector.py


def benchmark(model_path: str, task: str, runs: int = 20) -> float:
    """Returns average inference time in milliseconds over `runs` frames."""
    model     = YOLO(model_path, task=task)
    dummy     = np.random.randint(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    # Warmup
    for _ in range(3):
        model.predict(dummy, imgsz=INPUT_SIZE, verbose=False)
    # Timed runs
    t0 = time.perf_counter()
    for _ in range(runs):
        model.predict(dummy, imgsz=INPUT_SIZE, verbose=False)
    elapsed = (time.perf_counter() - t0) / runs * 1000
    return elapsed


def export_fp32_ncnn():
    print("\n── Step 1: Export FP32 NCNN ─────────────────────────────")
    if os.path.exists(MODEL_NCNN_FP32):
        print(f"  Already exists: ./{MODEL_NCNN_FP32}/  (skipping)")
        return
    print("  Converting PyTorch → NCNN (FP32)...")
    model = YOLO(MODEL_PT)
    model.export(format="ncnn", imgsz=INPUT_SIZE)
    print(f"  Saved: ./{MODEL_NCNN_FP32}/")


def export_int8_ncnn():
    print("\n── Step 2: Export INT8 quantized NCNN ───────────────────")
    if os.path.exists(MODEL_NCNN_INT8):
        print(f"  Already exists: ./{MODEL_NCNN_INT8}/  (skipping)")
        return
    print("  Applying INT8 quantization + NCNN export...")
    print("  This converts FP32 weights → INT8 (4x smaller, faster on ARM)")
    model = YOLO(MODEL_PT)
    model.export(
        format = "ncnn",
        imgsz  = INPUT_SIZE,
        int8   = True,
    )
    # Ultralytics creates a folder named yolov8n_ncnn_model_int8
    generated = MODEL_NCNN_FP32 + "_int8"
    if os.path.exists(generated) and not os.path.exists(MODEL_NCNN_INT8):
        os.rename(generated, MODEL_NCNN_INT8)
    print(f"  Saved: ./{MODEL_NCNN_INT8}/")


def run_benchmarks():
    print("\n── Step 3: Benchmark comparison ─────────────────────────")
    results = {}

    print("  Benchmarking original PyTorch FP32...")
    try:
        results["PyTorch FP32"] = benchmark(MODEL_PT, "detect")
    except Exception as e:
        print(f"  PyTorch benchmark failed: {e}")

    if os.path.exists(MODEL_NCNN_FP32):
        print("  Benchmarking NCNN FP32...")
        try:
            results["NCNN FP32"] = benchmark(MODEL_NCNN_FP32, "detect")
        except Exception as e:
            print(f"  NCNN FP32 benchmark failed: {e}")

    if os.path.exists(MODEL_NCNN_INT8):
        print("  Benchmarking NCNN INT8...")
        try:
            results["NCNN INT8"] = benchmark(MODEL_NCNN_INT8, "detect")
        except Exception as e:
            print(f"  NCNN INT8 benchmark failed: {e}")

    if results:
        print()
        print("  Results (lower = faster):")
        print("  " + "─" * 42)
        baseline = results.get("PyTorch FP32")
        for name, ms in results.items():
            speedup = f"  ({baseline/ms:.1f}x faster)" if baseline and name != "PyTorch FP32" else ""
            print(f"  {name:<20} {ms:>7.1f} ms / frame{speedup}")
        print("  " + "─" * 42)

        best = min(results, key=results.get)
        print(f"\n  Best model: {best}  ({results[best]:.1f} ms/frame)")
        fps = 1000 / results[best]
        print(f"  Estimated FPS on Pi: ~{fps:.1f}")


def main():
    print("VisionGuide — Model Optimization")
    print("=================================")
    print(f"Base model : {MODEL_PT}")
    print(f"Input size : {INPUT_SIZE}×{INPUT_SIZE} px")
    print(f"Target     : Raspberry Pi ARM CPU")

    export_fp32_ncnn()
    export_int8_ncnn()
    run_benchmarks()

    print("\n✓ Optimization complete.")
    print("  run.py will automatically use the fastest available model.")
    print()


if __name__ == "__main__":
    main()