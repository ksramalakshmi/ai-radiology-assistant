# -------------------
# YOLOv8 Segmentation Training Script in Google Colab
# -------------------

from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

# While running in colab
import yaml
import tempfile
import zipfile
import os

# Path to your zip file
zip_path = '/content/yolo_seg.zip'

# Directory where you want to extract
extract_dir = '/content/data'

# Make sure the extraction directory exists
os.makedirs(extract_dir, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Unzipped {zip_path} to {extract_dir}")


# Dataset dict
datasets = {
    'path': '/content/data/yolo_seg',
    'train': 'train/images',
    'val': 'val/images',
    'test': 'val/images',
    'nc': 1,
    'names': {0: 'region'}
}

# Write to temporary YAML file
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(datasets, f)
    dataset = f.name

def set_backbone_trainability(model, mode="freeze_full", num_layers=0):
    """
    Control backbone freezing/unfreezing.

    Args:
        model: YOLO-style model with model.model.model as backbone.
        mode: str, one of:
            - "freeze_full"   → freeze entire backbone
            - "freeze_bottom" → freeze first `num_layers` layers
            - "unfreeze_top"  → unfreeze last `num_layers` layers
        num_layers: int, number of layers to freeze/unfreeze (used only for freeze_bottom/unfreeze_top)

    Returns:
        model with updated requires_grad flags
    """
    backbone_layers = list(model.model.model)
    total_layers = len(backbone_layers)
    logger.info(f"Backbone has {total_layers} layers")

    if mode == "freeze_full":
        logger.info("Freezing full backbone")
        for layer in backbone_layers:
            for p in layer.parameters():
                p.requires_grad = False

    elif mode == "freeze_bottom":
        logger.info(f"Freezing bottom {num_layers} layers")
        for i in range(min(num_layers, total_layers)):
            for p in backbone_layers[i].parameters():
                p.requires_grad = False

    elif mode == "unfreeze_top":
        logger.info(f"Unfreezing top {num_layers} layers")
        for i in range(total_layers - num_layers, total_layers):
            for p in backbone_layers[i].parameters():
                p.requires_grad = True

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Log stats
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    return model

# -----------------
# Load model
# -----------------
model = YOLO("yolov8s-seg.pt")
# model = set_backbone_trainability(model=model, mode="unfreeze_top", num_layers=5)

# -----------------
# Train with early stopping + logging
# -----------------
model.train(
    data=dataset,
    epochs=30,
    imgsz=640,
    batch=1,
    device="cuda:0",
    workers=4,
    name="yolo_seg_kneexray-13_08-r1",
    optimizer="Adam",
    save=True,                # save checkpoints
    project="runs/segment",   # logging directory
    exist_ok=True,
    verbose=True
)
