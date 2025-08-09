import cv2
import numpy as np

def extract_features(image):
    try:
        resized = cv2.resize(image, (100, 100))
        features = resized.flatten()
        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None
