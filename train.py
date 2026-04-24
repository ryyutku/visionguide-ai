import os
import torch
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
 
 
BASE_MODEL = "yolov8n.pt"
DATA_YAML = "data/processed/data.yaml"
EPOCHS = 100
BATCH_SIZE = 32
IMAGE_SIZE = 640
LEARNING_RATE = 0.02
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
OUTPUT_DIR = "runs/train"
EXPERIMENT_NAME = "yolo_finetune_v1"


# loads a pretrained yolo checkpoint from ultralytics or a local path
def load_pretrained_model(model_path):
    if not os.path.exists(model_path) and not model_path.endswith(".pt"):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = YOLO(model_path)
    print(f"Loaded model: {model_path}")
    print(f"Model type: {type(model)}")
    return model


# reads the data.yaml and prints a summary so you can confirm classes before training
def verify_dataset_config(data_yaml_path):
    with open(data_yaml_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Dataset path: {config.get('path')}")
    print(f"Number of classes: {config.get('nc')}")
    print(f"Class names: {config.get('names')}")
    return config


# checks if a gpu is available and falls back to cpu if not
def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("No GPU found, using CPU. Training will be slow.")
    return device


# freezes the backbone layers so only the head is trained during initial fine-tuning
def freeze_backbone(model, num_freeze_layers=10):
    frozen = 0
    for i, (name, param) in enumerate(model.model.named_parameters()):
        if i < num_freeze_layers:
            param.requires_grad = False
            frozen += 1
    print(f"Frozen {frozen} backbone layers")
    return model


# unfreezes all layers for the second stage of fine-tuning
def unfreeze_all_layers(model):
    unfrozen = 0
    for param in model.model.parameters():
        param.requires_grad = True
        unfrozen += 1
    print(f"Unfrozen {unfrozen} total layers")
    return model


# runs training with the given hyperparameters and saves checkpoints to output dir
def train_model(model, data_yaml, epochs, batch_size, img_size, lr, device, output_dir, name):
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=lr,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        device=device,
        project=output_dir,
        name=name,
        save=True,
        save_period=10,
        patience=20,
        workers=4,
        verbose=True,
        plots=True,
        amp=True,
        cos_lr=True,
        label_smoothing=0.1
    )
    return results


# runs validation on the best checkpoint and prints metrics
def evaluate_model(model, data_yaml, img_size, device):
    metrics = model.val(
        data=data_yaml,
        imgsz=img_size,
        device=device,
        verbose=True
    )
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    return metrics


# saves the final model weights in both .pt and onnx formats
def export_model(model, output_dir, name):
    weights_path = Path(output_dir) / name / "weights" / "best.pt"
    if weights_path.exists():
        export_model_obj = YOLO(str(weights_path))
        export_model_obj.export(format="onnx", imgsz=IMAGE_SIZE)
        print(f"Model exported to ONNX at {weights_path.parent}")
    else:
        print(f"Could not find best.pt at {weights_path}")


# parses command line arguments so training params can be overridden at runtime
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO model")
    parser.add_argument("--model", type=str, default=BASE_MODEL)
    parser.add_argument("--data", type=str, default=DATA_YAML)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--freeze", action="store_true", help="Freeze backbone initially")
    parser.add_argument("--name", type=str, default=EXPERIMENT_NAME)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = get_device()
    verify_dataset_config(args.data)

    model = load_pretrained_model(args.model)

    if args.freeze:
        print("Phase 1: Training with frozen backbone")
        model = freeze_backbone(model)
        train_model(model, args.data, epochs=20, batch_size=args.batch,
                    img_size=args.imgsz, lr=args.lr * 0.1, device=device,
                    output_dir=OUTPUT_DIR, name=args.name + "_phase1")

        print("Phase 2: Fine-tuning all layers")
        model = unfreeze_all_layers(model)

    results = train_model(
        model, args.data, args.epochs, args.batch,
        args.imgsz, args.lr, device, OUTPUT_DIR, args.name
    )

    evaluate_model(model, args.data, args.imgsz, device)
    export_model(model, OUTPUT_DIR, args.name)
