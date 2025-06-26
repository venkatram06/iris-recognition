import cv2
import numpy as np
from preprocessing import preprocess_image
from segmentation import detect_iris_and_pupil
from normalization import get_circle_parameters
from normalization import normalize_iris
from feature_extraction import extract_features
from matching import match_features
import os
import tempfile

# Paths
RAW_IMG = "data/raw/webcam_eye.jpg"
SEG_IMG = "data/processed/segmented_iris.jpg"
NORM_IMG = "data/processed/normalized_iris.jpg"
FEAT_IMG = "data/processed/feature_map_live.jpg"
REF_IMG = "data/processed/feature_map.jpg"  # Reference image

def process_frame_and_match():
    preprocess_image(RAW_IMG)
    detect_iris_and_pupil(RAW_IMG, SEG_IMG)
    pupil_center, pupil_radius, iris_radius = get_circle_parameters(SEG_IMG)
    normalize_iris(RAW_IMG, pupil_center, pupil_radius, iris_radius)
    extract_features(NORM_IMG)
    score = match_features(REF_IMG, FEAT_IMG)
    return score

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("🎥 Press 's' to capture eye and match. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        cv2.imshow("Webcam Iris Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Save frame to disk
            eye_frame = frame.copy()
            cv2.imwrite(RAW_IMG, eye_frame)

            try:
                score = process_frame_and_match()
                print(f"✅ Score: {score:.4f}", end=' ')
                if score > 0.95:
                    print("→ MATCH ✅")
                else:
                    print("→ NO MATCH ❌")
            except Exception as e:
                print("⚠️ Error:", e)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    main()
