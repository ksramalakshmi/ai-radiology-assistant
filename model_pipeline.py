# Executed in Google Colab
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
from ultralytics import YOLO
from tqdm import tqdm

# ----------------------------
# Load models
# ----------------------------
seg_model = YOLO("/content/drive/MyDrive/best.pt")  # YOLOv8 segmentation model with 1 class
device = "cuda" if torch.cuda.is_available() else "cpu"

classifier = models.efficientnet_b0(pretrained=False)
classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, 3)
classifier.load_state_dict(torch.load("/content/drive/MyDrive/best_model.pt", map_location=device))
classifier.to(device)
classifier.eval()

input_size = 224
classifier_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class_map = {0: "normal", 1: "osteopenia", 2: "osteoporosis"}

# ----------------------------
# Validation dataset path
# ----------------------------
val_images_dir = "/content/drive/MyDrive/yolo_seg/val/images"

results_list = []

for img_file in tqdm(os.listdir(val_images_dir), desc="Processing YOLO validation images"):
    if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    img_path = os.path.join(val_images_dir, img_file)

    # Read image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run YOLO segmentation
    seg_results = seg_model(img_rgb)

    masks = seg_results[0].masks.xy if seg_results[0].masks is not None else []

    # If no mask, skip
    if len(masks) == 0:
        continue

    # Sort masks from left to right (based on centroid x-coordinate)
    mask_centroids = []
    for mask in masks:
        mask_np = np.array(mask, dtype=np.int32)
        M = cv2.moments(mask_np)
        cx = int(M["m10"]/M["m00"]) if M["m00"] != 0 else 0
        mask_centroids.append((cx, mask_np))
    mask_centroids.sort(key=lambda x: x[0])  # left to right

    # Assign orientation: left, right
    if len(mask_centroids) == 1:
        # Single knee, assume right
        masks_to_process = [("right", mask_centroids[0][1])]
    else:
        masks_to_process = [("left", mask_centroids[0][1]), ("right", mask_centroids[1][1])]

    # Process each mask separately
    for orientation, mask_np in masks_to_process:
        # Create grey 128 background
        masked_img = np.ones_like(img_rgb, dtype=np.uint8) * 128
        cv2.fillPoly(masked_img, [mask_np], color=(255,255,255))

        # Convert to PIL and preprocess
        masked_pil = Image.fromarray(masked_img).resize((input_size, input_size))
        img_tensor = classifier_transform(masked_pil).unsqueeze(0).to(device)

        # Classifier prediction
        with torch.no_grad():
            outputs = classifier(img_tensor)
            pred_class = torch.argmax(outputs, dim=1).item()
            pred_label = class_map[pred_class]

        results_list.append({
            "image_path": img_path,
            "orientation": orientation,
            "predicted_class": pred_label
        })

# ----------------------------
# Print sample results
# ----------------------------
for r in results_list[:10]:
    print(r)
