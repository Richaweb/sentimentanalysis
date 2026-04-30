# app/app.py
import streamlit as st
import sys
import os

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from predict import predict_migraine, get_category_labels

st.title('Migraine Symptom Classifier')

# Get dropdown options from predict.py
location_options, character_options = get_category_labels()

# User input form
age = st.number_input('Age', min_value=0)
duration = st.number_input('Duration (hours)', min_value=0.0)
frequency = st.number_input('Frequency', min_value=0)

location = st.selectbox('Location', location_options)
character = st.selectbox('Character', character_options)

nausea = st.selectbox('Nausea', ['No', 'Yes'])
visual = st.selectbox('Visual Symptoms', ['No', 'Yes'])

if st.button('Predict'):
    nausea_value = 1 if nausea == 'Yes' else 0
    visual_value = 1 if visual == 'Yes' else 0

    prediction = predict_migraine(
        age=age,
        duration=duration,
        frequency=frequency,
        location=location,
        character=character,
        nausea=nausea_value,
        visual=visual_value
    )

    st.write(f'### Predicted Migraine Type: {prediction}')
