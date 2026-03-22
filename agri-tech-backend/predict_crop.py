# predict_crop.py (updated)
import os
import joblib
import pickle
import numpy as np
from crop_report_generator import CropReportGenerator
from fastapi import BackgroundTasks
# Base backend directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model directory
MODELS_DIR = os.path.join(BASE_DIR, "crop-recommendation", "crop-recommendation-models")

# Processed data directory
DATA_DIR = os.path.join(BASE_DIR, "crop-recommendation", "crop-recommendation-processed_data")

# Reports directory
REPORTS_DIR = os.path.join(BASE_DIR, "report_crop")

# Feature names
FEATURE_NAMES = [
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Temperature",
    "Humidity",
    "Soil_pH",
    "Rainfall"
]

# Load stacked ensemble model
model_path = os.path.join(MODELS_DIR, "stacked_ensembel.pkl")
model = joblib.load(model_path)

# Load label encoder
data_path = os.path.join(DATA_DIR, "preprocessed_data.pkl")


with open(data_path, "rb") as f:
    data = pickle.load(f)

label_encoder = data["label_encoder"]
report_generator = CropReportGenerator(
    model=model,
    label_encoder=label_encoder,
    feature_names=FEATURE_NAMES,
    reports_dir=REPORTS_DIR
)

print("✅ Crop model and report generator loaded successfully")


def predict_crop(data, generate_report=True):
    try:
        # Convert input into model format
        features = np.array([[
            data["nitrogen"],
            data["phosphorus"],
            data["potassium"],
            data["temperature"],
            data["humidity"],
            data["ph"],
            data["rainfall"]
        ]])

        # Model prediction
        prediction = model.predict(features)[0]

        # Convert encoded label to crop name
        crop = label_encoder.inverse_transform([prediction])[0]

        # Confidence score
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities.max())
        
        # Get top 5 predictions
        top_5_indices = np.argsort(probabilities)[-5:][::-1]
        top_5_crops = label_encoder.inverse_transform(top_5_indices).tolist()
        top_5_probs = probabilities[top_5_indices].tolist()

        result = {
            "crop": crop,
            "confidence": confidence,
            "probabilities": {
                crop_name: float(prob) 
                for crop_name, prob in zip(label_encoder.classes_, probabilities)
            },
            "top_5_predictions": [
                {"crop": crop_name, "probability": float(prob)}
                for crop_name, prob in zip(top_5_crops, top_5_probs)
            ]
        }
        
        # Generate report if requested
        if generate_report:
            try:
                report_path = report_generator.generate_prediction_report(data, result)
                result["report_path"] = "/report/latest"
                result["report_id"] = "latest"
            except Exception as e:
                print(f"Error generating report: {e}")
                result["report_error"] = str(e)

        return result

    except Exception as e:
        return {
            "error": str(e)
        }