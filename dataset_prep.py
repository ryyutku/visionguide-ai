import os
import shutil
import random
import cv2
import yaml
import numpy as np
from pathlib import Path


RAW_IMAGES_DIR = "data/raw/images"
RAW_LABELS_DIR = "data/raw/labels"
OUTPUT_DIR = "data/processed"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
IMAGE_SIZE = 640


# creates the folder structure yolo expects: images/train, images/val, images/test
def create_directory_structure(base_dir):
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)
    print(f"Directory structure created at {base_dir}")


# reads all image paths and shuffles them before splitting
def load_and_shuffle_dataset(images_dir):
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_paths = [
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in valid_extensions
    ]
    random.shuffle(image_paths)
    return image_paths


# splits the dataset list into train/val/test based on the ratios defined above
def split_dataset(image_paths, train_ratio, val_ratio):
    total = len(image_paths)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = image_paths[:train_end]
    val = image_paths[train_end:val_end]
    test = image_paths[val_end:]

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


# copies images and their matching label files into the correct split folder
def copy_files_to_split(image_paths, labels_dir, output_dir, split_name):
    copied = 0
    missing_labels = 0

    for img_path in image_paths:
        label_path = Path(labels_dir) / (img_path.stem + ".txt")

        dest_img = Path(output_dir) / "images" / split_name / img_path.name
        shutil.copy2(img_path, dest_img)

        if label_path.exists():
            dest_label = Path(output_dir) / "labels" / split_name / label_path.name
            shutil.copy2(label_path, dest_label)
            copied += 1
        else:
            missing_labels += 1

    print(f"[{split_name}] Copied: {copied}, Missing labels: {missing_labels}")


# resizes images to the target size if they dont match already
def resize_image_if_needed(image_path, target_size):
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    if h != target_size or w != target_size:
        img_resized = cv2.resize(img, (target_size, target_size))
        cv2.imwrite(str(image_path), img_resized)
    return True


# writes the data.yaml file that yolo reads to understand class names and paths
def generate_data_yaml(output_dir, class_names):
    yaml_content = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names
    }
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"data.yaml written to {yaml_path}")


# checks that every image in a split has a corresponding label file
def validate_dataset(output_dir, split_name):
    images_dir = Path(output_dir) / "images" / split_name
    labels_dir = Path(output_dir) / "labels" / split_name

    image_stems = {p.stem for p in images_dir.iterdir()}
    label_stems = {p.stem for p in labels_dir.iterdir()}

    unmatched = image_stems - label_stems
    if unmatched:
        print(f"[{split_name}] WARNING: {len(unmatched)} images without labels")
    else:
        print(f"[{split_name}] All images have labels")


# normalizes bounding box coordinates from pixel space to 0-1 range
def normalize_bbox(x_min, y_min, x_max, y_max, img_w, img_h):
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height


if __name__ == "__main__":
    random.seed(42)

    class_names = ["cat", "dog", "person", "car", "bicycle"]

    create_directory_structure(OUTPUT_DIR)

    image_paths = load_and_shuffle_dataset(RAW_IMAGES_DIR)
    train_paths, val_paths, test_paths = split_dataset(image_paths, TRAIN_SPLIT, VAL_SPLIT)

    copy_files_to_split(train_paths, RAW_LABELS_DIR, OUTPUT_DIR, "train")
    copy_files_to_split(val_paths, RAW_LABELS_DIR, OUTPUT_DIR, "val")
    copy_files_to_split(test_paths, RAW_LABELS_DIR, OUTPUT_DIR, "test")

    generate_data_yaml(OUTPUT_DIR, class_names)

    for split in ["train", "val", "test"]:
        validate_dataset(OUTPUT_DIR, split)
