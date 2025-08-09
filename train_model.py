import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

dataset_path = "dataset"
labels = []
data = []

print("🔍 Loading images...")
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if os.path.isdir(label_path):
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image_resized = cv2.resize(image, (100, 100))
                features = image_resized.flatten()
                data.append(features)
                labels.append(label)

# Convert to NumPy arrays
X = np.array(data)
y = np.array(labels)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Training model...")
model = SVC(kernel='linear')
model.fit(X_train, y_train)

print("✅ Model trained!")

# Evaluate
print("\n📊 Classification Report:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/blood_group_model.pkl")
print("💾 Model saved to model/blood_group_model.pkl")
