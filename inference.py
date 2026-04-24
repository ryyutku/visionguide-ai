import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict


MODEL_PATH = "runs/train/yolo_finetune_v1/weights/best.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640
OUTPUT_DIR = "runs/inference"
CLASS_NAMES = ["cat", "dog", "person", "car", "bicycle"]

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255)
]


# loads the fine-tuned model weights from the given path
def load_model(model_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD):
    model = YOLO(model_path)
    model.conf = conf
    model.iou = iou
    print(f"Model loaded from {model_path}")
    print(f"Confidence threshold: {conf}, IoU threshold: {iou}")
    return model


# runs inference on a single image and returns raw result objects
def predict_single(model, image_path, img_size=IMAGE_SIZE):
    results = model.predict(
        source=str(image_path),
        imgsz=img_size,
        verbose=False
    )
    return results[0]


# runs batched inference on a folder of images for efficiency
def predict_batch(model, images_dir, img_size=IMAGE_SIZE, batch_size=16):
    image_paths = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
    all_results = []

    for i in range(0, len(image_paths), batch_size):
        batch = [str(p) for p in image_paths[i:i + batch_size]]
        results = model.predict(source=batch, imgsz=img_size, verbose=False)
        all_results.extend(results)
        print(f"Processed {min(i + batch_size, len(image_paths))} / {len(image_paths)}")

    return all_results, image_paths


# draws bounding boxes and labels onto the image for visualization
def draw_detections(image, result, class_names=CLASS_NAMES):
    annotated = image.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return annotated

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())

        color = COLORS[cls_id % len(COLORS)]
        label = f"{class_names[cls_id]} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - label_size[1] - 4),
                      (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated


# converts a result object into a list of plain dicts for easy json serialization
def result_to_dict(result, image_name):
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            detections.append({
                "image": image_name,
                "class_id": int(box.cls[0]),
                "class_name": CLASS_NAMES[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": [x1, y1, x2, y2]
            })
    return detections


# computes precision, recall and f1 for a single class given tp fp fn counts
def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


# calculates intersection over union between two bounding boxes
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-9)


# matches predicted boxes to ground truth boxes using iou and returns counts per class
def evaluate_predictions(pred_boxes, gt_boxes, iou_thresh=0.5):
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    matched_gt = set()

    for pred in pred_boxes:
        cls = pred["class_id"]
        matched = False
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            if gt["class_id"] == cls:
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou >= iou_thresh:
                    stats[cls]["tp"] += 1
                    matched_gt.add(j)
                    matched = True
                    break
        if not matched:
            stats[cls]["fp"] += 1

    for j, gt in enumerate(gt_boxes):
        if j not in matched_gt:
            stats[gt["class_id"]]["fn"] += 1

    return stats


# saves all detection results to a json file for downstream analysis
def save_results_json(all_detections, output_path):
    with open(output_path, "w") as f:
        json.dump(all_detections, f, indent=2)
    print(f"Results saved to {output_path}")


# prints a per-class summary table with precision recall f1 and detection counts
def print_metrics_summary(per_class_stats, class_names=CLASS_NAMES):
    print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-" * 70)
    for cls_id, stats in per_class_stats.items():
        p, r, f1 = compute_metrics(stats["tp"], stats["fp"], stats["fn"])
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        print(f"{name:<15} {p:<12.4f} {r:<12.4f} {f1:<12.4f} "
              f"{stats['tp']:<8} {stats['fp']:<8} {stats['fn']:<8}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model(MODEL_PATH)

    test_images_dir = "data/processed/images/test"
    results, image_paths = predict_batch(model, test_images_dir)

    all_detections = []
    for result, img_path in zip(results, image_paths):
        dets = result_to_dict(result, img_path.name)
        all_detections.extend(dets)

        image = cv2.imread(str(img_path))
        if image is not None:
            annotated = draw_detections(image, result)
            out_path = Path(OUTPUT_DIR) / img_path.name
            cv2.imwrite(str(out_path), annotated)

    save_results_json(all_detections, os.path.join(OUTPUT_DIR, "detections.json"))
    print(f"Inference complete. {len(all_detections)} total detections.")
