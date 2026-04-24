import cv2
import random
import numpy as np
from pathlib import Path


# augmentation probability settings applied per image during preprocessing
FLIP_PROB = 0.5
ROTATE_PROB = 0.3
BRIGHTNESS_PROB = 0.4
BLUR_PROB = 0.2
NOISE_PROB = 0.2
CUTOUT_PROB = 0.3
MOSAIC_PROB = 0.5
MAX_ROTATION_ANGLE = 15
CUTOUT_PATCHES = 3
CUTOUT_RATIO = 0.1


# randomly flips the image and mirrors all bounding boxes along the x axis
def horizontal_flip(image, bboxes):
    if random.random() < FLIP_PROB:
        image = cv2.flip(image, 1)
        flipped_bboxes = []
        for bbox in bboxes:
            cls, x_center, y_center, w, h = bbox
            x_center = 1.0 - x_center
            flipped_bboxes.append((cls, x_center, y_center, w, h))
        return image, flipped_bboxes
    return image, bboxes


# rotates the image by a small random angle and adjusts bbox coordinates accordingly
def random_rotation(image, bboxes, max_angle=MAX_ROTATION_ANGLE):
    if random.random() > ROTATE_PROB:
        return image, bboxes

    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    return rotated, bboxes


# randomly adjusts brightness and contrast to simulate different lighting conditions
def color_jitter(image):
    if random.random() > BRIGHTNESS_PROB:
        return image

    brightness = random.uniform(0.6, 1.4)
    contrast = random.uniform(0.6, 1.4)

    image = image.astype(np.float32)
    image = image * contrast + (brightness - 1) * 128
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


# applies a slight gaussian blur to simulate out of focus or low res cameras
def random_blur(image):
    if random.random() > BLUR_PROB:
        return image
    kernel_size = random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


# adds gaussian noise to the image to improve robustness to sensor noise
def add_gaussian_noise(image):
    if random.random() > NOISE_PROB:
        return image
    std = random.uniform(5, 25)
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# randomly blacks out small rectangular patches to force the model to learn partial features
def cutout(image, num_patches=CUTOUT_PATCHES, ratio=CUTOUT_RATIO):
    if random.random() > CUTOUT_PROB:
        return image

    h, w = image.shape[:2]
    patch_h = int(h * ratio)
    patch_w = int(w * ratio)

    result = image.copy()
    for _ in range(num_patches):
        y = random.randint(0, h - patch_h)
        x = random.randint(0, w - patch_w)
        result[y:y + patch_h, x:x + patch_w] = 0

    return result


# combines four images into one mosaic tile, used to increase context diversity
def mosaic_augmentation(images, bboxes_list, output_size=640):
    if len(images) < 4:
        return images[0], bboxes_list[0]

    half = output_size // 2
    canvas = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    new_bboxes = []

    positions = [(0, 0), (half, 0), (0, half), (half, half)]
    offsets = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]

    for i in range(4):
        img = cv2.resize(images[i], (half, half))
        x_off, y_off = positions[i]
        canvas[y_off:y_off + half, x_off:x_off + half] = img

        for bbox in bboxes_list[i]:
            cls, xc, yc, bw, bh = bbox
            xc = (xc * 0.5) + offsets[i][0]
            yc = (yc * 0.5) + offsets[i][1]
            bw = bw * 0.5
            bh = bh * 0.5
            new_bboxes.append((cls, xc, yc, bw, bh))

    return canvas, new_bboxes


# clips any bounding boxes that go outside the image boundary after augmentation
def clip_bboxes(bboxes, min_size=0.01):
    clipped = []
    for bbox in bboxes:
        cls, xc, yc, w, h = bbox
        x_min = max(0.0, xc - w / 2)
        y_min = max(0.0, yc - h / 2)
        x_max = min(1.0, xc + w / 2)
        y_max = min(1.0, yc + h / 2)

        new_w = x_max - x_min
        new_h = y_max - y_min

        if new_w > min_size and new_h > min_size:
            new_xc = (x_min + x_max) / 2
            new_yc = (y_min + y_max) / 2
            clipped.append((cls, new_xc, new_yc, new_w, new_h))

    return clipped


# runs the full augmentation pipeline on a single image and its labels
def data_augment_sample(image, bboxes):
    image, bboxes = horizontal_flip(image, bboxes)
    image, bboxes = random_rotation(image, bboxes)
    image = color_jitter(image)
    image = random_blur(image)
    image = add_gaussian_noise(image)
    image = cutout(image)
    bboxes = clip_bboxes(bboxes)
    return image, bboxes


# processes a whole directory and writes augmented copies next to originals
def augment_dataset(images_dir, labels_dir, output_images_dir, output_labels_dir, copies=2):
    Path(output_images_dir).mkdir(parents=True, exist_ok=True)
    Path(output_labels_dir).mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
    print(f"Found {len(image_paths)} images to augment")

    for img_path in image_paths:
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        with open(label_path, "r") as f:
            bboxes = []
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    bboxes.append((int(parts[0]), float(parts[1]),
                                   float(parts[2]), float(parts[3]), float(parts[4])))

        for copy_idx in range(copies):
            aug_img, aug_bboxes = augment_sample(image.copy(), bboxes.copy())

            out_img_name = f"{img_path.stem}_aug{copy_idx}{img_path.suffix}"
            Change_img_name = f"{img_path.stem}_aug{copy_idx}{img_path.suffix}"
            out_label_name = f"{img_path.stem}_aug{copy_idx}.txt"

            cv2.imwrite(str(Path(output_images_dir) / out_img_name), aug_img)

            with open(Path(output_labels_dir) / out_label_name, "w") as f:
                for bbox in aug_bboxes:
                    f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")

    print("Augmentation complete")


if __name__ == "__main__":
    augment_dataset(
        images_dir="data/processed/images/train",
        labels_dir="data/processed/labels/train",
        output_images_dir="data/augmented/images/train",
        output_labels_dir="data/augmented/labels/train",
        copies=3
    )
