# src/train.py
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load data
df = pd.read_csv('../data/migrainedata.csv')

# Selected features
selected_features = ['Age', 'Duration', 'Frequency', 'Location', 'Character', 'Nausea', 'Visual']

X = df[selected_features]
y = df['Type']

# Encode categorical features
le_location = LabelEncoder()
X['Location'] = le_location.fit_transform(X['Location'])

le_character = LabelEncoder()
X['Character'] = le_character.fit_transform(X['Character'])
print(os.curdir)
# Save encoders for later use
joblib.dump(le_location, '../models/le_location.pkl')
joblib.dump(le_character, '../models/le_character.pkl')

# Scale numerical features
scaler = StandardScaler()
X[['Age', 'Duration', 'Frequency']] = scaler.fit_transform(X[['Age', 'Duration', 'Frequency']])

# Save scaler
joblib.dump(scaler, '../models/scaler.pkl')

# Handle imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, '../models/migraine_model.pkl')

print('Model and preprocessors saved successfully!')
