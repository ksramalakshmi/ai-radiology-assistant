import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
from ultralytics import YOLO
from fpdf import FPDF
import datetime
import ollama
import re
import base64

# ----------------------------
# Paths and Directories
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")                # YOLO segmentation
CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pt")  # EfficientNet
UPLOAD_DIR = "uploads"
PDF_DIR = "reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

# ----------------------------
# Device
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ----------------------------
# Load Models
# ----------------------------
# YOLO segmentation
seg_model = YOLO(MODEL_PATH)

# EfficientNet classifier
classifier = models.efficientnet_b0(pretrained=False)
classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, 3)
classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
classifier.to(device)
classifier.eval()

input_size = 224
classifier_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class_map = {0: "normal", 1: "osteopenia", 2: "osteoporosis"}


# ----------------------------
# Segmentation + Classification
# ----------------------------
def run_segmentation_classifier(image_path: str):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    seg_results = seg_model(img_rgb)
    masks = seg_results[0].masks.xy if seg_results[0].masks is not None else []

    if len(masks) == 0:
        return None, None

    # Sort masks left to right
    mask_centroids = []
    for mask in masks:
        mask_np = np.array(mask, dtype=np.int32)
        M = cv2.moments(mask_np)
        cx = int(M["m10"]/M["m00"]) if M["m00"] != 0 else 0
        mask_centroids.append((cx, mask_np))
    mask_centroids.sort(key=lambda x: x[0])

    if len(mask_centroids) == 1:
        masks_to_process = [("right", mask_centroids[0][1])]
    else:
        masks_to_process = [("left", mask_centroids[0][1]), ("right", mask_centroids[1][1])]

    results = {}
    overlay_img = img_rgb.copy()

    for orientation, mask_np in masks_to_process:
        masked_img = np.ones_like(img_rgb, dtype=np.uint8) * 128
        cv2.fillPoly(masked_img, [mask_np], color=(255, 255, 255))

        masked_pil = Image.fromarray(masked_img).resize((input_size, input_size))
        img_tensor = classifier_transform(masked_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = classifier(img_tensor)
            pred_class = torch.argmax(outputs, dim=1).item()
            pred_label = class_map[pred_class]

        results[orientation] = pred_label

        # Draw overlay
        cv2.polylines(overlay_img, [mask_np], isClosed=True, color=(255, 0, 0), thickness=3)
        M = cv2.moments(mask_np)
        cx = int(M["m10"]/M["m00"]) if M["m00"] != 0 else 0
        cy = int(M["m01"]/M["m00"]) if M["m00"] != 0 else 0
        cv2.putText(overlay_img, f"{orientation}: {pred_label}", (cx - 50, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    overlay_path = os.path.join(UPLOAD_DIR, os.path.basename(image_path).replace(".jpg", "_overlay.jpg"))
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

    return results, overlay_path


# ----------------------------
# QA + PDF Generation
# ----------------------------
def generate_report_pdf(predictions: dict):
    system_msg = "You are a professional radiology report generator. Respond in only one concise sentence. Avoid references or citations."

    def query_llm(user_content: str):
        text = ollama.chat(
            model="medllama2",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content}
            ],
        )["message"]["content"].strip()
        text = re.sub(r"\[\d+\]", "", text)
        return text.strip()

    # Right knee
    q_right = f"Patient ID: {predictions['patient_id']}. The right knee classification is '{predictions['right_knee']}'. Write one concise professional sentence describing it."
    right_report = query_llm(q_right)

    # Left knee
    q_left = f"Patient ID: {predictions['patient_id']}. The left knee classification is '{predictions['left_knee']}'. Write one concise professional sentence describing it."
    left_report = query_llm(q_left)

    # Overall impression
    q_summary = f"Patient ID: {predictions['patient_id']}. Based on the right and left knee findings: Right: {right_report} Left: {left_report}. Write one concise professional sentence summarizing the overall knee condition."
    overall_summary = query_llm(q_summary)

    # Recommendation
    q_recommendation = f"Patient ID: {predictions['patient_id']}. Based on the overall summary: '{overall_summary}', give one short clinical recommendation in one sentence."
    recommendation = query_llm(q_recommendation)

    # Assemble report
    report_text = f"""Patient Name: {predictions['patient_name']}
Patient ID: {predictions['patient_id']}

Findings:
- Right Knee: {right_report}

- Left Knee: {left_report}

Overall Impression: {overall_summary}

Recommendation: {recommendation}"""

    # Save PDF
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Knee X-ray Radiology Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 8, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, report_text)

    pdf_file_name = os.path.join(PDF_DIR, f"{predictions['patient_id']}_knee_report.pdf")
    pdf.output(pdf_file_name)

    return pdf_file_name


st.set_page_config(layout="wide")
st.title("AI Radiology Assistant")

# --------------------
# Patient Details Form
# --------------------
with st.form("patient_form"):
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Name", value="John Doe")
        patient_id = st.text_input("Patient ID", value="P12345")
    with col2:
        technologist = st.selectbox(
            "Technologist",
            ["Tech001", "Tech002", "Tech003", "Tech004"],
            index=0
        )

    submitted = st.form_submit_button("Save Details")
    if submitted:
        st.success(f"Patient details saved")

# --------------------
# Upload & Processing
# --------------------
uploaded_file = st.file_uploader("Upload a knee X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save temp image
    img = Image.open(uploaded_file).convert("RGB")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(temp_file.name)

    # Run segmentation + classification
    results, overlay_path = run_segmentation_classifier(temp_file.name)

    if results is None:
        st.error("No knees detected in the X-ray.")
    else:
        # Use dynamic patient details
        predictions = {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "technologist": technologist,
            "right_knee": results.get("right", "not detected"),
            "left_knee": results.get("left", "not detected"),
        }

        # Divide layout into two halves
        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.image(overlay_path, caption="Predicted condition", use_container_width=True)

        with right_col:
            # Generate PDF automatically
            pdf_file = generate_report_pdf(predictions)

            # Show PDF inline (scrollable, full height)
            with open(pdf_file, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")

            pdf_display = f"""
            <div style="height:100vh;">
                <iframe src="data:application/pdf;base64,{base64_pdf}"
                        width="100%" height="100%" style="border:none;"></iframe>
            </div>
            """
            st.markdown(pdf_display, unsafe_allow_html=True)

# --------------------
# Sticky Footer Section
# --------------------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 8px;
        font-size: 14px;
        color: gray;
        border-top: 1px solid #e6e6e6;
        z-index: 100;
    }
    </style>
    <div class="footer">
        © 2025 AI Radiology Assistant. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
