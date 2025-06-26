import cv2
import numpy as np

def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {path}")
    return image

def denoise_image(image: np.ndarray, method: str = "gaussian") -> np.ndarray:
    if method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == "median":
        return cv2.medianBlur(image, 5)
    else:
        raise ValueError("Unsupported method.")

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def preprocess_image(image_path: str) -> np.ndarray:
    image = read_image(image_path)
    denoised = denoise_image(image)
    enhanced = enhance_contrast(denoised)
    return enhanced

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    input_path = "data/raw/sample_eye.jpg"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    processed = preprocess_image(input_path)

    output_path = os.path.join(output_dir, "preprocessed.jpg")
    cv2.imwrite(output_path, processed)

    original = read_image(input_path)
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title("Preprocessed")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
