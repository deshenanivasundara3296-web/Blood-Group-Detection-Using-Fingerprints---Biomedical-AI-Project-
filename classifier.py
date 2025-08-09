import joblib
import numpy as np

# Load the trained model
model = joblib.load("model/blood_group_model.pkl")

def predict_blood_group(features):
    try:
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]
        return prediction
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unknown"
