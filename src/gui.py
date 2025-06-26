import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import os
import shutil

from preprocessing import *
from segmentation import *
from normalization import *
from feature_extraction import *
from matching import *

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_DIR, "..", "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "..", "data", "processed")
REFERENCE_IMG = os.path.join(PROCESSED_DIR, "feature_map.jpg")  # previously saved

class IrisApp:
    def __init__(self, root):
        self.root = root
        root.title("Iris Recognition GUI")

        self.label = Label(root, text="Upload an Eye Image", font=("Arial", 14))
        self.label.pack(pady=10)

        self.upload_btn = Button(root, text="Choose Image", command=self.upload_image)
        self.upload_btn.pack(pady=5)

        self.result_label = Label(root, text="", font=("Arial", 12), fg="green")
        self.result_label.pack(pady=10)

        self.image_label = Label(root)
        self.image_label.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select Eye Image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file_path:
            # Copy uploaded image to raw folder
            dest_path = os.path.join(RAW_DIR, "uploaded_eye.jpg")
            shutil.copy(file_path, dest_path)

            # Run recognition pipeline
            try:
                preprocess_image(dest_path)
                detect_iris_and_pupil(dest_path, os.path.join(PROCESSED_DIR, "segmented_iris.jpg"))
                pupil_center, pupil_radius, iris_radius = get_circle_parameters(os.path.join(PROCESSED_DIR, "segmented_iris.jpg"))
                normalize_iris(dest_path, pupil_center, pupil_radius, iris_radius)
                extract_features(os.path.join(PROCESSED_DIR, "normalized_iris.jpg"))

                # Compare with reference
                test_img = os.path.join(PROCESSED_DIR, "feature_map.jpg")
                score = match_features(REFERENCE_IMG, test_img)

                match_text = f"Cosine Similarity Score: {score:.4f}"
                if score > 0.95:
                    match_text += "\n✅ Match Found"
                else:
                    match_text += "\n❌ No Match"

                self.result_label.config(text=match_text, fg="green" if score > 0.95 else "red")
                self.show_image(dest_path)
            except Exception as e:
                self.result_label.config(text=f"Error: {str(e)}", fg="red")

    def show_image(self, path):
        img = Image.open(path)
        img = img.resize((300, 300))
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo

if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    root = tk.Tk()
    app = IrisApp(root)
    root.mainloop()
