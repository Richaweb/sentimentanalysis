# src/predict.py
import joblib
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'migraine_model.pkl')
LOCATION_ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'le_location.pkl')
CHARACTER_ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'le_character.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

# Load the model and preprocessors
model = joblib.load(MODEL_PATH)
le_location = joblib.load(LOCATION_ENCODER_PATH)
le_character = joblib.load(CHARACTER_ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_migraine(age, duration, frequency, location, character, nausea, visual):
    # Encode user input
    encoded_location = le_location.transform([location])[0]
    encoded_character = le_character.transform([character])[0]

    # Scale numeric input
    scaled_values = scaler.transform([[age, duration, frequency]])

    # Prepare input for the model
    input_features = [
        scaled_values[0][0],  # Scaled Age
        scaled_values[0][1],  # Scaled Duration
        scaled_values[0][2],  # Scaled Frequency
        encoded_location,
        encoded_character,
        nausea,
        visual
    ]

    # Make prediction
    prediction = model.predict([input_features])[0]
    return prediction

# Get available category labels for dropdowns
def get_category_labels():
    return le_location.classes_, le_character.classes_
