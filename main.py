import os
from utils.image_loader import load_image
from preprocess import preprocess_image
from feature_extraction import extract_features
from classifier import predict_blood_group

def main():
    image_path = "data/captured_fingerprint.png"

    if not os.path.exists(image_path):
        print(f"Image not found at: {image_path}")
        return

    image = load_image(image_path)

    if image is not None:
        print("✅ Image loaded.")
        processed = preprocess_image(image)
        features = extract_features(processed)
        if features is not None:
            result = predict_blood_group(features)
            print(f"🩸 Predicted Blood Group: {result}")
        else:
            print("❌ Feature extraction failed.")
    else:
        print("❌ Failed to load image.")

if __name__ == "__main__":
    main()
