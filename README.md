# Bone Fracture Detection and Classification using Deep Learning and Computer Vision

## Overview
Bone fractures are among the most common medical injuries and often require expert radiologists for accurate diagnosis and classification from X-ray images. Manual interpretation can be time-consuming and may lead to inconsistencies in high-volume clinical environments.

This project presents an AI-powered web application that performs automated bone fracture detection and classification using Deep Learning and Computer Vision techniques. The system validates uploaded images, detects fractures, classifies fracture types, and generates explainable visualizations for educational and diagnostic support.
The application is built using Flask, PyTorch, OpenCV, and cloud-ready deployment architecture.

# Research Publication

## Paper Title
An Explainable Deep Learning Framework for Automated Bone Fracture Detection and Classification

## Conference
2026 International Conference on Emerging Systems and Intelligent Computing (ESIC)

## Authors
Vinay P  
Pruthvi M Kadri  
Mehak Sharma  
Moulya Suvarna  
Pooja

📄 IEEE Xplore Publication:  
[An Explainable Deep Learning Framework for Automated Bone Fracture Detection and Classification](https://ieeexplore.ieee.org/document/11496076?source=document-share&dld=Z21haWwuY29t)

# Features
- X-ray image validation before inference
- Binary fracture detection
- Multi-class fracture classification
- Explainable AI visualizations using Grad-CAM
- Edge-enhanced hybrid visualizations using Canny Edge Detection
- Confidence score generation
- Report generation and storage
- Flask-based interactive web application
- Cloud-ready backend architecture
- Modular deployment structure for scalable environments

# System Workflow
text
User Uploads X-ray Image
            ↓
X-ray Validation Module
            ↓
Binary Fracture Detection
            ↓
Multi-Class Fracture Classification
            ↓
Grad-CAM and Hybrid Visualization
            ↓
Result Storage and Report Generation

# Fracture Classes Supported
The system supports classification of the following fracture types:
1. Avulsion Fracture
2. Comminuted Fracture
3. Fracture Dislocation
4. Greenstick Fracture
5. Hairline Fracture
6. Impacted Fracture
7. Longitudinal Fracture
8. Oblique Fracture
9. Pathological Fracture
10. Spiral Fracture
11. Unknown Fracture

# Technology Stack

| Component | Technology |
|---|---|
| Backend Framework | Flask |
| Deep Learning Framework | PyTorch |
| Image Processing | OpenCV, NumPy |
| Frontend | HTML, CSS, JavaScript |
| Deployment Architecture | Gunicorn, Render-ready setup |
| Optional Database | MongoDB / SQLite |
| Model Format | .pt / .pth |

# Explainable AI and Visualization
The project integrates Explainable AI techniques to improve transparency and interpretability of model predictions.
## Visualization Modules
### Grad-CAM Visualization
Highlights important regions influencing the model prediction.

### Canny Edge Detection
Enhances fracture boundaries and edge structures.

### Hybrid Visualization
Combines Grad-CAM overlays with edge detection outputs for educational analysis and visual interpretability.

# Application Screenshots

## Home Page
<img width="1911" height="910" alt="image" src="https://github.com/user-attachments/assets/2f40f6e8-3f06-4aaf-91b9-13e563a333c8" />

## Signup Page
<img width="1524" height="895" alt="Screenshot (1828)" src="https://github.com/user-attachments/assets/8709ba26-ab38-4c5c-8d89-44aeff2d39d0" />

## otp verify Page
<img width="1785" height="513" alt="Screenshot (1826)" src="https://github.com/user-attachments/assets/de8b9d19-f2d6-44e2-89a7-8a2229a3d497" />

## login Page
<img width="1768" height="546" alt="Screenshot (1825)" src="https://github.com/user-attachments/assets/d3a10d64-5c1a-4734-80c9-2d73837ae66f" />

## X-ray Upload Interface
<img width="1871" height="835" alt="Screenshot (1821)" src="https://github.com/user-attachments/assets/66b6fb94-abfc-4eb5-8ea9-e60a6741f41c" />

## Fracture Prediction Output
<img width="1858" height="910" alt="Screenshot (1822)" src="https://github.com/user-attachments/assets/5e719489-6f8e-4c85-88a5-7982c4535998" />

## PDF Report
<img width="638" height="827" alt="Screenshot (1824)" src="https://github.com/user-attachments/assets/3885d09c-a021-4a66-929c-b083c19255a7" />

# Model Pipeline

## Step 1: X-ray Validation
The uploaded image is first validated to determine whether it is a legitimate bone X-ray image.

## Step 2: Binary Fracture Detection
The validated image is passed through a binary classification model to detect the presence or absence of fractures.

## Step 3: Multi-Class Fracture Classification
If a fracture is detected, the image is classified into one of the supported fracture categories.

## Step 4: Explainable Visualization
Grad-CAM and Canny Edge visualizations are generated for interpretability and educational support.

## Step 5: Result Storage
Prediction results and generated outputs are stored for future reference and reporting.

# Folder Structure
text
Bone-Fracture-Detection-and-Classification/
│
├── Homees/
│   ├── main.py
│   ├── wsgi.py
│   ├── download_models.py
│   ├── requirements.txt
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   ├── models/
│   │   └── uploads/
│   └── templates/
│
├── .gitignore
└── README.md

# Installation Guide
## Clone Repository
bash
git clone <repository-link>


## Navigate to Project Directory
bash
cd Homees/


## Install Dependencies
bash
pip install -r requirements.txt


## Run Application
bash
python main.py

# Deployment Architecture
The project architecture was designed to support scalable cloud deployment using Flask, Gunicorn, and Render-compatible backend configuration.
During development, deployment experiments highlighted practical challenges associated with large deep learning model hosting on low-resource cloud environments.

The application includes:
- WSGI deployment structure
- Cloud-ready backend organization
- Model download pipeline support
- Deployment-compatible project modularization

Future optimization strategies may include:
- Model quantization
- ONNX/TensorRT optimization
- Lightweight inference pipelines
- GPU-enabled deployment services
# Model Storage Strategy
Large model files are excluded from GitHub using .gitignore and handled separately to maintain repository efficiency.
## Ignored Files
text
*.pt
*.pth
.env
static/models/
static/uploads/
__pycache__/
.venv/
# Key Contributions
- AI-powered fracture classification system
- Explainable AI visualization integration
- Hybrid Grad-CAM and edge-enhanced analysis
- Flask backend and UI integration
- Deployment architecture preparation
- Automated model management workflow
- Educational visualization framework
# Learning Outcomes
This project provided practical experience in:
- Deep Learning model development
- Medical image processing
- Explainable AI techniques
- OpenCV-based preprocessing
- Flask backend development
- Cloud deployment architecture concepts
- Understanding deployment limitations in low-resource environments
- GitHub project structuring and deployment workflows
# Future Enhancements
- 3D fracture detection pipeline
- Mobile deployment using TensorFlow Lite
- REST API integration for hospital systems
- Real-time doctor feedback integration
- Continual learning framework
- Improved uncertainty estimation
- Multi-modal medical analysis

# Contributors
- Vinay P
- Pruthvi M Kadri
- Mehak Sharma
- Moulya Suvarna
- Pooja
