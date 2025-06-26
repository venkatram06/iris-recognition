import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

def load_feature_vector(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Flatten to 1D vector and normalize
    vector = image.flatten().astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def match_features(image1_path, image2_path):
    vec1 = load_feature_vector(image1_path)
    vec2 = load_feature_vector(image2_path)

    # Compute cosine similarity
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    return similarity

if __name__ == "__main__":
    img1 = "data/processed/feature_map.jpg"       # Replace with your first feature image
    img2 = "data/processed/feature_map.jpg"       # Replace with your second (or same for test)

    score = match_features(img1, img2)
    print(f"[✅] Cosine Similarity Score: {score:.4f}")

    if score > 0.95:
        print("✅ Iris Match: SAME PERSON")
    else:
        print("❌ Iris Match: DIFFERENT PERSON")
