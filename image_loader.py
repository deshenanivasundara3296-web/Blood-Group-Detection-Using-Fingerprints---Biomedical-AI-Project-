import cv2

def load_image(path):
    try:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
