# Iris Recognition System – DevifyX ML Assignment

This project implements an end-to-end Iris Recognition System with preprocessing, segmentation, normalization, feature extraction, and matching. It also supports real-time webcam-based recognition and liveness detection (blink detection).

## 🔧 Features Implemented

- ✅ Image Preprocessing  
- ✅ Iris Segmentation  
- ✅ Pupil & Limbic Boundary Detection  
- ✅ Normalization (Polar Unwrapping)  
- ✅ Feature Extraction (Gabor Filters)  
- ✅ Feature Matching (Cosine Similarity)  
- ✅ GUI for Testing  
- ✅ Real-Time Webcam Recognition  
- ✅ Liveness Detection (Blink)  
- ✅ Organized Pipeline with Modular Code  

## 🖥️ Tech Stack

- Python 3.8+  
- OpenCV  
- NumPy  
- Matplotlib  
- MediaPipe  
- TensorFlow  
- Tkinter (GUI)  

## 📁 Project Structure

iris-recognition/
├── data/
│ ├── raw/
│ └── processed/
├── src/
│ ├── preprocessing.py
│ ├── segmentation.py
│ ├── normalization.py
│ ├── feature_extraction.py
│ ├── matching.py
│ ├── gui_app.py
│ ├── webcam_recognition.py
│ └── liveness_detection.py
├── requirements.txt
├── README.md
└── report.md

📺 Demo Video
🎥 Click to Watch the Demo on Google Drive(https://drive.google.com/open?id=1Hqa4ozrgwriw6GXro34R8i-b-K97zXc3)

✅ Demo Includes:
Image Preprocessing & Segmentation

Iris Normalization & Feature Extraction

Cosine Similarity Matching

⚙️ Additional Functionalities (Implemented in Code):
GUI Interface → python src/gui_app.py

Real-time Webcam Iris Capture → python src/webcam_recognition.py

Blink-based Liveness Detection → python src/liveness_detection.py   

## 📦 Setup Instructions

```bash
pip install -r requirements.txt
python src/gui_app.py

✅ How to Run
Run preprocessing and segmentation

Normalize the iris

Extract features

Match features

Optional: Run GUI or webcam recognition

👨‍💻 Author
Venkatramireddy
