import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def detect_iris_and_pupil(image_path, save_path="data/processed/segmented_iris.jpg"):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image not found")

    # Denoise the image
    blurred = cv2.medianBlur(image, 5)

    # Detect circles (pupil and iris)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=100, param2=30, minRadius=20, maxRadius=100)

    # Convert to color for drawing
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i, circle in enumerate(circles[0, :2]):  # Draw only top 2 circles
            center = (circle[0], circle[1])
            radius = circle[2]
            if i == 0:
                label = "Pupil"
                color = (0, 255, 0)
            else:
                label = "Iris"
                color = (255, 0, 0)
            cv2.circle(output, center, radius, color, 2)
            cv2.putText(output, label, (center[0]-20, center[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        print("No circles detected.")

    # Save image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, output)
    print(f"[✅] Segmented image saved at: {save_path}")

    # Show image
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Iris & Pupil Detection")
    plt.axis("off")
    plt.show()

# Run the function
if __name__ == "__main__":
    detect_iris_and_pupil("data/raw/sample2_eye.jpg")
