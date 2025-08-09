import cv2
import os

def capture_fingerprint(save_path="data/captured_fingerprint.png"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return None

    print("Press 's' to save the fingerprint, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Capture Fingerprint", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            # Convert to grayscale and save
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, gray)
            print(f"Fingerprint saved to: {save_path}")
            break
        elif key == ord('q'):
            print("Capture cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return save_path
