import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def normalize_iris(image_path, pupil_center, pupil_radius, iris_radius,
                   radial_res=64, angular_res=360):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image not found")

    cy, cx = pupil_center
    r_pupil = pupil_radius
    r_iris = iris_radius

    # Generate polar coordinate grid
    theta = np.linspace(0, 2 * np.pi, angular_res)
    r = np.linspace(0, 1, radial_res)

    r_matrix, theta_matrix = np.meshgrid(r, theta)
    radius = r_matrix * (r_iris - r_pupil) + r_pupil

    x = cx + radius * np.cos(theta_matrix)
    y = cy + radius * np.sin(theta_matrix)

    x = np.clip(x, 0, image.shape[1] - 1).astype(np.float32)
    y = np.clip(y, 0, image.shape[0] - 1).astype(np.float32)

    normalized = cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR)

    output_path = "data/processed/normalized_iris.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, normalized)
    print(f"[✅] Normalized iris saved at: {output_path}")

    plt.imshow(normalized, cmap='gray')
    plt.title("Normalized Iris")
    plt.axis("off")
    plt.show()


def get_circle_parameters(segmented_image_path):
    image = cv2.imread(segmented_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Segmented image not found")

    blurred = cv2.medianBlur(image, 5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=100, param2=30, minRadius=20, maxRadius=100)

    if circles is None or len(circles[0]) < 2:
        raise ValueError("Could not detect both pupil and iris circles.")

    circles = np.uint16(np.around(circles[0, :2]))  # first 2 circles
    pupil = circles[0]
    iris = circles[1]

    # Show detected circles
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i, c in enumerate([pupil, iris]):
        center = (c[0], c[1])
        radius = c[2]
        color = (0, 255, 0) if i == 0 else (255, 0, 0)
        label = "Pupil" if i == 0 else "Iris"
        cv2.circle(output, center, radius, color, 2)
        cv2.putText(output, label, (center[0]-20, center[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Detected Pupil and Iris Circles")
    plt.axis("off")
    plt.show()

    pupil_center = (pupil[1], pupil[0])  # (y, x)
    pupil_radius = pupil[2]
    iris_radius = iris[2]

    return pupil_center, pupil_radius, iris_radius


if __name__ == "__main__":
    segmented_image_path = "data/processed/segmented_iris.jpg"
    original_image_path = "data/raw/sample2_eye.jpg"  # Your new image

    pupil_center, pupil_radius, iris_radius = get_circle_parameters(segmented_image_path)
    normalize_iris(original_image_path, pupil_center, pupil_radius, iris_radius)
