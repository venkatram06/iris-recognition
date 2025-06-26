# Iris Recognition System – Report

## 📌 Objective
To build a complete Iris Recognition System that includes:
- Image preprocessing
- Iris segmentation
- Normalization
- Feature extraction
- Feature matching  
Bonus features: GUI, real-time recognition via webcam, and liveness detection.

---

## ✅ Core Features

1. **Preprocessing** – Noise reduction using median blur.
2. **Segmentation** – Detecting pupil and iris using Hough Circles.
3. **Normalization** – Rubber sheet model (polar unwrapping).
4. **Feature Extraction** – Gabor filter responses for texture-based features.
5. **Matching** – Cosine similarity for comparing iris features.
6. **Evaluation** – Manual testing and matching with different images.

---

## 🎁 Bonus Features

- GUI application (Tkinter)
- Real-time recognition via webcam
- Liveness detection via blink detection (MediaPipe)

---

## 📊 Dataset

- Sample custom images used for training and testing
- Optional use of IITD iris dataset for cross-evaluation

---

## 🛠️ Tech Stack

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- TensorFlow
- MediaPipe
- Tkinter (GUI)

---

## ✅ Conclusion

All core requirements were implemented successfully.  
Bonus features like GUI, real-time matching, and liveness detection were also completed.  
Project is modular, reproducible, and ready for deployment.
