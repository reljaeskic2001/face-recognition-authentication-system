# Face Recognition Authentication System

A computer vision and machine learning system that performs face detection, face recognition, user authentication, and facial attribute analysis using YOLOv8, DeepFace, and facial embeddings.

---

## Overview

This project implements a complete end-to-end face authentication pipeline capable of:

* Detecting faces in images and webcam streams
* Generating facial embeddings using deep learning models
* Identifying known individuals from a facial database
* Authenticating users in real time
* Estimating facial attributes such as age, gender, emotion, and glasses detection
* Evaluating multiple face recognition approaches

The system combines modern face detection and face recognition techniques to create a practical authentication workflow while providing performance evaluation and visualization tools.

---

## Features

### Face Detection

* YOLOv8-based face detector
* Single-person and group image support
* Real-time webcam detection
* Confidence score reporting

### Face Recognition

* DeepFace facial embeddings
* FaceNet512 backbone
* Cosine similarity matching
* Unknown-user rejection threshold

### Authentication System

* Real-time webcam authentication
* Identity verification
* Confidence score display
* Authentication dashboard

### Facial Attribute Analysis

* Age estimation
* Gender classification
* Emotion recognition
* Glasses detection

### Evaluation and Analysis

* 1-NN Cosine Similarity
* 3-NN Cosine Similarity
* Linear SVM
* RBF SVM
* Similarity heatmap visualization
* Backbone comparison across multiple models

---

## Technologies Used

* Python
* OpenCV
* YOLOv8
* DeepFace
* TensorFlow
* NumPy
* Scikit-Learn
* Pandas
* Matplotlib
* Machine Learning
* Computer Vision

---

## Project Structure

```text
face-recognition-authentication-system/
│
├── code/
│   ├── authenticate.py
│   ├── attributes.py
│   ├── build_synthetic_database.py
│   ├── detect_faces.py
│   ├── embeddings.py
│   ├── identify.py
│   ├── part2_yolo_outputs.py
│   ├── part3_backbone_comparison.py
│   └── part3_evaluation.py
│
├── database/
│   ├── synthetic identities
│   └── user/
│
├── outputs/
│   ├── part2/
│   ├── similarity_heatmap.png
│   └── evaluation results
│
├── screenshots/
│
├── face_db.pkl
├── group_photo.jpg
├── yolov8n-face.pt
├── requirements.txt
└── README.md
```

### Folder Descriptions

**code/**
Contains all source code for face detection, embedding generation, evaluation, authentication, and attribute extraction.

**database/**
Stores facial images used to build the recognition database. Includes synthetic identities generated for testing and a dedicated `user/` folder for personal authentication.

**outputs/**
Stores generated outputs including detection visualizations, evaluation results, classifier metrics, and similarity heatmaps.

**screenshots/**
Contains screenshots used for documentation and demonstrations.

**face_db.pkl**
Serialized facial embedding database generated from all identities.

**group_photo.jpg**
Multi-person image used to evaluate face detection performance.

**yolov8n-face.pt**
Pretrained YOLOv8 face detection model weights.

---

## Using Your Own Face

By default, the repository contains synthetic identities used for testing and evaluation.

To authenticate yourself, add approximately 10–15 images of your face to:

```text
database/user/
```

Recommended images should include:

* Front-facing images
* Left and right head angles
* Upward and downward head angles
* Different facial expressions
* Different lighting conditions
* Glasses and non-glasses images (if applicable)

Example:

```text
database/
└── user/
    ├── 00.jpg
    ├── 01.jpg
    ├── 02.jpg
    ├── ...
    └── 14.jpg
```

After adding your images, regenerate the facial embedding database before running the authentication system.

---

## Methodology

### 1. Dataset Creation

The project uses a synthetic facial dataset generated from AI-created identities.

For each identity, multiple image variations are automatically created:

* Original image
* Horizontally flipped image
* Brightened image
* Darkened image
* Zoomed image

These augmentations increase the number of training samples and improve robustness.

Users may also add their own identity through the `database/user/` folder.

### 2. Face Detection

YOLOv8 is used to detect faces within images and webcam streams.

The detector:

* Locates face bounding boxes
* Assigns confidence scores
* Supports both single-face and multi-face images

Detected faces are cropped and passed to the recognition stage.

### 3. Facial Embedding Generation

DeepFace with the FaceNet512 backbone is used to generate facial embeddings.

Each face is converted into a normalized numerical representation that captures unique facial characteristics.

Embeddings are stored inside:

```text
face_db.pkl
```

and later used for identification and authentication.

### 4. Face Identification

Given a query face:

1. Generate facial embedding
2. Compare against stored embeddings
3. Compute cosine similarity
4. Select the highest similarity score
5. Reject unknown identities below a threshold

This allows the system to identify known individuals while rejecting unknown users.

### 5. Attribute Extraction

Additional facial attributes are estimated using DeepFace:

* Age
* Gender
* Emotion

A lightweight computer vision method is also used to estimate:

* Presence of glasses

### 6. Authentication Pipeline

The authentication system combines:

* Webcam input
* Face detection
* Embedding generation
* Face identification
* Attribute analysis

The system displays:

* Authentication status
* Identity prediction
* Similarity score
* Age estimate
* Gender prediction
* Emotion prediction
* Glasses detection
* Real-time FPS

---

## Evaluation

The project evaluates multiple recognition methods:

### K-Nearest Neighbors

* 1-NN Cosine Similarity
* 3-NN Cosine Similarity

### Support Vector Machines

* Linear SVM
* RBF SVM

### Similarity Heatmap

A similarity matrix is generated to visualize relationships between identities and evaluate recognition quality.

### Backbone Comparison

The following face recognition backbones are compared:

* FaceNet512
* ArcFace
* VGG-Face

This allows performance comparison across different embedding models.

---



## How to Run

### 1. Clone Repository

```bash
git clone https://github.com/reljaeskic2001/face-recognition-authentication-system.git

cd face-recognition-authentication-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Synthetic Database (Optional)

```bash
cd code

python build_synthetic_database.py
```

This creates synthetic identities used for testing and evaluation.

### 4. Add Personal Images (Optional)

Place approximately 10–15 images of your face inside:

```text
database/user/
```

### 5. Generate Face Embeddings

```bash
python embeddings.py
```

This creates:

```text
face_db.pkl
```

### 6. Test Face Detection

```bash
python detect_faces.py
```

### 7. Generate Detection Outputs

```bash
python part2_yolo_outputs.py
```

Outputs are saved to:

```text
outputs/part2/
```

### 8. Run Evaluation

```bash
python part3_evaluation.py
```

### 9. Compare Recognition Backbones

```bash
python part3_backbone_comparison.py
```

### 10. Launch Authentication System

```bash
python authenticate.py
```

Press **ESC** to exit the application.

---

## Future Improvements

* Larger and more diverse datasets
* Real-time multi-person tracking
* Mobile deployment
* Cloud-based authentication
* Multi-factor authentication
* Liveness detection
* Improved robustness to lighting and occlusions

---

## Author

**Relja Eskic**

M.S. Computer Science
University of Denver
