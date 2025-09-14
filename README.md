# AI Radiology Assistant  

This repository contains a development prototype for an AI-powered radiology assistant that analyzes knee X-ray images.  
It combines **object segmentation** using YOLO, **bone density classification** using EfficientNet, and automated **report generation** in a Streamlit-based interface.  



## Overview  

The project demonstrates a complete pipeline starting from image upload, preprocessing, segmentation, classification, and automated report generation. It is intended as a foundation for extending AI-based clinical tools and workflows.  



## Pipeline  

1. **Image Upload & Preprocessing**  
   - Input X-ray images (`.jpg`, `.jpeg`, `.png`) are normalized and resized.  
   - Preprocessing ensures consistent input to both YOLO (for segmentation) and EfficientNet (for classification).  

2. **Segmentation (YOLO + SAM)**  
   - A YOLO model (`best.pt`) trained for knee localization is used to detect bounding boxes for left and right knees.  
   - The [Segment Anything Model (SAM, ViT-B)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) is used with YOLO outputs to refine segmentation masks.  
   - Masks are sorted left-to-right for orientation consistency.  

3. **Classification (EfficientNet-B0)**  
   - Segmented regions are passed into an EfficientNet-B0 model fine-tuned for three classes:  
     - Normal  
     - Osteopenia  
     - Osteoporosis  
   - Each knee is classified independently.  

4. **Report Generation**  
   - Findings, impressions, and recommendations are generated using a lightweight LLM integration.  
   - A PDF report is created with patient and technologist details, along with classification outcomes.  

5. **Streamlit Integration**  
   - Interactive web interface for uploading images and entering patient details.  
   - Two-column layout:  
     - Left: uploaded X-ray and overlay with predictions.  
     - Right: auto-generated PDF report shown inline with download option.  
   - Sticky footer for project credits.  



## Data Preprocessing  

- Images were standardized to RGB.  
- Resized to `224x224` for EfficientNet input.  
- Masks generated from YOLO + SAM outputs were applied to extract bone regions.  
- Data augmentation techniques were applied during training to improve model generalization.  



## Features  

- Upload knee X-ray images for analysis.  
- Automated segmentation of left and right knees using YOLO + SAM.  
- Classification into clinically relevant bone density categories.  
- PDF report generation with findings, impression, and recommendation.  
- Patient and technologist details integrated into the workflow.  
- Inline PDF viewer with download functionality.  



## Installation  

Clone the repository and install dependencies:  

```bash
git clone https://github.com/your-username/ai-radiology-assistant.git
cd ai-radiology-assistant
pip install -r requirements.txt
```
 
 
## Run the Application

To start the Streamlit app, run:

```bash
streamlit run streamlit_app.py
```



## Project Structure

``` bash
├── models/                 # Pretrained YOLO and EfficientNet weights
├── uploads/                # Temporary uploaded images
├── reports/                # Generated PDF reports
├── streamlit_app.py        # Main Streamlit app
├── requirements.txt        # Python dependencies
└── README.md
```



## Next Steps

- Expand classification beyond three categories.
- Improve segmentation for varied X-ray qualities.
- Add support for additional imaging modalities.
- Optimize LLM integration for faster inference.


