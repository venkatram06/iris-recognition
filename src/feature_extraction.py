import cv2
import numpy as np
import os

def extract_features(image_path, output_path="data/processed/feature_map_live.jpg"):
    """
    Applies Gabor filters to extract discriminative features from a normalized iris image.
    
    Parameters:
        image_path (str): Path to the normalized iris image.
        output_path (str): Path to save the resulting feature map.
    """
    # Load the normalized iris image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Normalized iris image not found.")

    # Create a bank of Gabor filters at 0°, 45°, 90°, 135°
    filters = []
    ksize = 31  # kernel size
    for theta in np.arange(0, np.pi, np.pi / 4):
        kernel = cv2.getGaborKernel((ksize, ksize), sigma=4.0, theta=theta,
                                    lambd=10.0, gamma=0.5, psi=0)
        filters.append(kernel)

    # Apply all filters and keep the maximum response
    feature_map = np.zeros_like(image, dtype=np.float32)
    for kernel in filters:
        filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
        feature_map = np.maximum(feature_map, filtered)

    # Normalize to 0–255
    feature_map -= feature_map.min()
    feature_map /= feature_map.max()
    feature_map = (feature_map * 255).astype(np.uint8)

    # Save the feature map
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, feature_map)
    print(f"[✅] Feature map saved at: {output_path}")

# ✅ Add this block to run directly
if __name__ == "__main__":
    extract_features("data/processed/normalized_iris.jpg")
