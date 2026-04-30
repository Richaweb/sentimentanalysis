import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def predict_migraine(age, duration, frequency, location, character, nausea, visual):
    model = joblib.load('migraine_model.pkl')
    location = le_location.transform([location])[0]
    character = le_character.transform([character])[0]
    scaled_values = scaler.transform([[age, duration, frequency]])

    input_features = [
        scaled_values[0][0],  # Scaled Age
        scaled_values[0][1],  # Scaled Duration
        scaled_values[0][2],  # Scaled Frequency
        location,
        character,
        nausea,
        visual
    ]

    prediction = model.predict([input_features])
    return prediction[0]