"""
VisualAid Indoor Detection - Full Training Pipeline
====================================================
Uses the FREE COCO 2017 dataset (via FiftyOne) to fine-tune a YOLOv8 model
for indoor obstacle detection to assist visually impaired users.


Usage:
r"venv\Scripts\Activate"
pip install -r requirements.txt

    python visualaid_pipeline.py
"""

import os
import logging
from pathlib import Path

# -----------------------------------------------------------------------------
#  CONFIG - edit these values to customise the pipeline
# -----------------------------------------------------------------------------
CONFIG = {
    # Indoor objects relevant to blind navigation.
    # All names must exactly match COCO class names.
    "target_classes": [
        "chair",
        "couch",
        "bed",
        "dining table",
    ], 

    # Number of training images to download per class.
    # 200 = fast demo | 500+ = better accuracy | 1000+ = production quality
    "train_samples_per_class": 120,

    # Number of validation images to download per class.
    "val_samples_per_class": 20,

    # Where to save the prepared dataset.
    "dataset_dir": "visualaid_dataset",

    # YOLOv8 variant: n(ano) | s(mall) | m(edium) | l(arge) | x(large)
    # yolov8n is recommended - smallest and fastest for Raspberry Pi.
    "yolo_model": "yolov8n.pt",

    # Training hyper-parameters
    "epochs":   20,
    "imgsz":    416,
    "batch":    16,     # reduce to 8 if you run out of RAM
    "lr0":      0.01,
    "patience": 10,     # early stopping
    "freeze":   10,     # freeze first N backbone layers
    "device": "cpu",
    # Export formats for Raspberry Pi deployment
    # Options: "ncnn", "tflite", "onnx"
    "export_formats": ["ncnn", "tflite", "onnx"],

    # Output folders
    "project_name": "runs",
    "run_name":     "visualaid_indoor",
}
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ==============================================================================
#  STEP 1 - Download COCO dataset via FiftyOne (free, no API key needed)
# ==============================================================================

def download_dataset(cfg):
    """
    Download COCO 2017 train and validation splits filtered to the
    target classes using FiftyOne's dataset zoo.
    Returns (train_dataset, val_dataset)
    """
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        raise ImportError(
            "FiftyOne is not installed.\n"
            "Run:  pip install fiftyone"
        )

    classes = cfg["target_classes"]
    train_n = cfg["train_samples_per_class"] * len(classes)
    val_n   = cfg["val_samples_per_class"]   * len(classes)

    log.info("Downloading COCO 2017 TRAIN split (%d samples max)...", train_n)
    train_ds = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        classes=classes,
        max_samples=train_n,
        dataset_name="visualaid_coco_train",
        overwrite=True,
    )

    log.info("Downloading COCO 2017 VAL split (%d samples max)...", val_n)
    val_ds = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=classes,
        max_samples=val_n,
        dataset_name="visualaid_coco_val",
        overwrite=True,
    )

    log.info(
        "Downloaded %d train images | %d val images",
        len(train_ds), len(val_ds)
    )
    return train_ds, val_ds


# ==============================================================================
#  STEP 2 - Export to YOLOv8 folder structure
# ==============================================================================

def export_to_yolo(train_ds, val_ds, cfg):
    """
    Export FiftyOne datasets to YOLOv8-compatible folder layout and
    write dataset.yaml. Returns path to dataset.yaml.

    Folder structure created:
        visualaid_dataset/
            train/images/   <- jpg files
            train/labels/   <- yolo .txt annotation files
            val/images/
            val/labels/
            dataset.yaml
    """
    try:
        import fiftyone as fo
    except ImportError:
        raise ImportError("FiftyOne is not installed. pip install fiftyone")

    classes    = cfg["target_classes"]
    output_dir = Path(cfg["dataset_dir"])

    for split, ds in [("train", train_ds), ("val", val_ds)]:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        log.info("Exporting %s split to %s...", split, split_dir)
        ds.export(
            export_dir=str(split_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            classes=classes,
            overwrite=True,
        )

    # FiftyOne exports images into <split>/data/ by default
    # Some versions use images/ instead - check both
    train_img_dir = output_dir / "train" / "data"
    val_img_dir   = output_dir / "val"   / "data"

    if not train_img_dir.exists():
        train_img_dir = output_dir / "train" / "images"
        val_img_dir   = output_dir / "val"   / "images"

    yaml_path = output_dir / "dataset.yaml"
    yaml_lines = [
        "# Auto-generated by visualaid_pipeline.py",
        "path: " + str(output_dir.resolve()),
        "train: " + str(train_img_dir.relative_to(output_dir)),
        "val:   " + str(val_img_dir.relative_to(output_dir)),
        "",
        "nc: " + str(len(classes)),
        "names: " + str(classes),
        "",
    ]
    yaml_path.write_text("\n".join(yaml_lines))
    log.info("dataset.yaml written to %s", yaml_path)

    return yaml_path


# ==============================================================================
#  STEP 3 - Fine-tune YOLOv8
# ==============================================================================

def train_model(yaml_path, cfg):
    """
    Load a pretrained YOLOv8 nano checkpoint and fine-tune it on the
    indoor obstacle dataset. Returns path to best weights.
    """
    import torch
    # Check for GPU and print it
    device = "0" if torch.cuda.is_available() else "cpu"
    log.info(f"TRAINING DEVICE DETECTED: {device}")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is not installed.\n"
            "Run:  pip install ultralytics"
        )

    log.info("Loading pretrained model: %s", cfg["yolo_model"])
    model = YOLO(cfg["yolo_model"])

    log.info("Starting fine-tuning...")
    model.train(
        data=str(yaml_path),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        lr0=cfg["lr0"],
        patience=cfg["patience"],
        freeze=cfg["freeze"],
        project=cfg["project_name"],
        name=cfg["run_name"],
        exist_ok=True,
        # Data augmentation settings
        hsv_h=0.015,   # hue shift
        hsv_s=0.7,     # saturation shift
        hsv_v=0.4,     # brightness shift
        fliplr=0.5,    # horizontal flip probability
        mosaic=1.0,    # mosaic augmentation (combines 4 images)
        mixup=0.1,     # mixup augmentation
    )

    best_weights = Path(cfg["project_name"]) / "detect" / cfg["run_name"] / "weights" / "best.pt"
    log.info("Training complete. Best weights: %s", best_weights)
    return best_weights


# ==============================================================================
#  STEP 4 - Validate and print metrics
# ==============================================================================

def validate_model(weights_path, yaml_path):
    """
    Run validation set through the fine-tuned model and log key metrics.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return

    if not weights_path.exists():
        log.warning("Weights not found at %s - skipping validation.", weights_path)
        return

    log.info("Running validation...")
    model   = YOLO(str(weights_path))
    metrics = model.val(data=str(yaml_path))

    log.info("--- Validation Results ---")
    log.info("  mAP@50    : %.4f  (main accuracy benchmark)",       metrics.box.map50)
    log.info("  mAP@50-95 : %.4f  (stricter benchmark)",            metrics.box.map)
    log.info("  Precision : %.4f  (how often detections are right)", metrics.box.mp)
    log.info("  Recall    : %.4f  (how many real objects found)",    metrics.box.mr)
    log.info("--------------------------")

    return metrics


# ==============================================================================
#  STEP 5 - Export for Raspberry Pi edge deployment
# ==============================================================================

def export_for_edge(weights_path, cfg):
    """
    Export the trained model to formats optimised for Raspberry Pi.

    NCNN   - fastest inference on ARM CPUs, no runtime needed
    TFLite - int8 quantised, great for Pi Zero and Pi 4
    ONNX   - universal fallback, works on all platforms
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return

    if not weights_path.exists():
        log.warning("Weights not found - skipping export.")
        return

    model = YOLO(str(weights_path))

    export_settings = {
        "ncnn":   {"format": "ncnn"},
        "tflite": {"format": "tflite", "int8": True},
        "onnx":   {"format": "onnx",   "simplify": True},
    }

    for fmt in cfg["export_formats"]:
        if fmt not in export_settings:
            log.warning("Unknown export format: %s - skipping.", fmt)
            continue
        log.info("Exporting to %s...", fmt.upper())
        try:
            model.export(**export_settings[fmt], imgsz=cfg["imgsz"])
            log.info("%s export complete.", fmt.upper())
        except Exception as exc:
            log.warning("%s export failed: %s", fmt.upper(), exc)


# ==============================================================================
#  STEP 6 - Run inference with accessibility alerts
# ==============================================================================

def run_inference(weights_path, source, conf=0.45):
    """
    Run the model on an image, video, or webcam and print
    human-readable audio alerts designed for visually impaired users.

    Args:
        source : file path | URL | 0 for webcam
        conf   : confidence threshold (0 to 1)

    Examples:
        run_inference(weights, "room_photo.jpg")
        run_inference(weights, 0)   # live webcam feed
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return

    HIGH_PRIORITY   = {"chair", "couch", "dining table", "bed", }

    model   = YOLO(str(weights_path))
    results = model.predict(source=source, conf=conf, save=True, stream=True)

    for frame_result in results:
        alerts = []

        for box in frame_result.boxes:
            label      = frame_result.names[int(box.cls)]
            confidence = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Estimate rough position in frame
            frame_w    = frame_result.orig_shape[1]
            box_centre = (x1 + x2) / 2

            if box_centre < frame_w * 0.33:
                position = "on your left"
            elif box_centre > frame_w * 0.66:
                position = "on your right"
            else:
                position = "ahead of you"

            # Assign urgency based on object type
            if label in HIGH_PRIORITY:
                urgency = "WARNING"
            else:
                urgency = "INFO"

            alerts.append(
                f"{urgency}: {label} {position} (confidence: {confidence:.0%})"
            )

        if alerts:
            print("\n" + "-" * 50)
            for a in alerts:
                print(a)
        else:
            print("Clear path detected.")


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    # --- ADD THE GPU CHECK HERE ---
    import torch
    print("\n" + "="*50)
    print(f"DEBUG: Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"DEBUG: Current Device: {torch.cuda.get_device_name(0)}")
        try:
            x = torch.rand(5, 3).cuda()
            print("DEBUG: Successfully moved data to GPU!")
        except Exception as e:
            print(f"DEBUG: Error moving data to GPU: {e}")
    else:
        print("DEBUG: GPU NOT DETECTED. The script will use CPU (Slooooow).")
    print("="*50 + "\n")

    cfg = CONFIG
    dataset_path = Path(cfg["dataset_dir"])
    # Adjust this path based on the "detect" folder bug we found earlier
    weights_path = Path(cfg["project_name"]) / "detect" / cfg["run_name"] / "weights" / "best.pt"

    log.info("================================================")
    log.info("  VisualAid Indoor Detection - Training Pipeline")
    log.info("================================================")

    # --- STEP 1 & 2: Data Preparation ---
    if not dataset_path.exists():
        log.info("Dataset not found. Downloading and exporting...")
        train_ds, val_ds = download_dataset(cfg)
        yaml_path = export_to_yolo(train_ds, val_ds, cfg)
    else:
        log.info(f"Dataset already exists at {dataset_path}. Skipping download/export.")
        yaml_path = dataset_path / "dataset.yaml"

    # --- STEP 3: Training ---
    if not weights_path.exists():
        log.info("No trained weights found. Starting training...")
        best_weights = train_model(yaml_path, cfg)
    else:
        log.info(f"Trained model found at {weights_path}. Skipping training.")
        best_weights = weights_path

    # --- STEP 4 & 5: Validation & Export ---
    # These are fast, but you can wrap them in checks too if you want
    validate_model(best_weights, yaml_path)
    export_for_edge(best_weights, cfg)

    log.info("================================================")
    log.info("  Pipeline complete!")
    log.info("  Best weights : %s", best_weights)
    log.info("  Edge exports : %s", best_weights.parent)
    log.info("================================================")

    # Optional: test inference on a sample image - uncomment to use:
    # run_inference(best_weights, "test_image.jpg")
    # run_inference(best_weights, 0)  # webcam


if __name__ == "__main__":
    main()
