import cv2
import numpy as np
import os
import random

labels = ["A+", "B+", "O+", "AB+"]
output_dir = "dataset"

def generate_image(label, idx):
    # Create a 100x100 grayscale dummy fingerprint-like image
    image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    # Add a visible circle to simulate fingerprint pattern
    cv2.circle(image, (random.randint(30, 70), random.randint(30, 70)), 10, (255,), -1)
    
    folder = os.path.join(output_dir, label)
    os.makedirs(folder, exist_ok=True)
    
    path = os.path.join(folder, f"fp{idx}.png")
    cv2.imwrite(path, image)
    print(f"✅ Saved: {path}")

def main():
    for label in labels:
        for i in range(10):  # 10 dummy images per group
            generate_image(label, i)

if __name__ == "__main__":
    main()
