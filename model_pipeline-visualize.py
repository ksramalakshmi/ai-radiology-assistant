# Executed in Google Colab and outputs saved in folder -> pipeine_outputs
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
seg_model = YOLO("/content/drive/MyDrive/best.pt")
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
# Paths
# ----------------------------
val_images_dir = "/content/drive/MyDrive/yolo_seg/val/images"
output_dir = "/content/drive/MyDrive/pipeline_outputs"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Process all validation images
# ----------------------------
for img_file in tqdm(os.listdir(val_images_dir), desc="Processing YOLO validation images"):
    if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(val_images_dir, img_file)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display_img = img_rgb.copy()

    # YOLO segmentation
    seg_results = seg_model(img_rgb)
    masks = seg_results[0].masks.xy if seg_results[0].masks is not None else []

    if len(masks) == 0:
        continue

    # Sort masks left -> right
    mask_centroids = []
    for mask in masks:
        mask_np = np.array(mask, dtype=np.int32)
        M = cv2.moments(mask_np)
        cx = int(M["m10"]/M["m00"]) if M["m00"] != 0 else 0
        mask_centroids.append((cx, mask_np))
    mask_centroids.sort(key=lambda x: x[0])

    # Assign orientation
    if len(mask_centroids) == 1:
        masks_to_process = [("right", mask_centroids[0][1])]
    else:
        masks_to_process = [("left", mask_centroids[0][1]), ("right", mask_centroids[1][1])]

    # Process each knee
    for orientation, mask_np in masks_to_process:
        # Grey 128 background + overlay mask
        masked_img = np.ones_like(img_rgb, dtype=np.uint8) * 128
        cv2.fillPoly(masked_img, [mask_np], color=(255,255,255))

        # Preprocess for classifier
        masked_pil = Image.fromarray(masked_img).resize((input_size, input_size))
        img_tensor = classifier_transform(masked_pil).unsqueeze(0).to(device)

        # Classifier prediction
        with torch.no_grad():
            outputs = classifier(img_tensor)
            pred_class = torch.argmax(outputs, dim=1).item()
            pred_label = class_map[pred_class]

        # Annotate display image
        cv2.polylines(display_img, [mask_np], isClosed=True, color=(255,0,0), thickness=4)
        M = cv2.moments(mask_np)
        cx = int(M["m10"]/M["m00"]) if M["m00"] != 0 else mask_np[0,0]
        cy = int(M["m01"]/M["m00"]) if M["m00"] != 0 else mask_np[0,1]
        cv2.putText(display_img, f"{orientation}: {pred_label}", (cx-50, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

    # Save annotated image
    output_path = os.path.join(output_dir, img_file)
    cv2.imwrite(output_path, cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
