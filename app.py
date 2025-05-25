import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model and feature columns
with open('models/heart_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

def user_input_features():
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Sex', ['F', 'M'])
    cp = st.selectbox('Chest Pain Type', ['ATA', 'ASY', 'NAP', 'TA'])
    resting_bp = st.number_input('Resting Blood Pressure', 80, 200, 120)
    chol = st.number_input('Cholesterol', 100, 600, 200)
    fasting_bs = st.selectbox('Fasting Blood Sugar (0 = <=120 mg/dl, 1 = >120 mg/dl)', ['0', '1'])
    restecg = st.selectbox('Resting ECG Results', ['LVH', 'Normal', 'ST'])
    max_hr = st.number_input('Max Heart Rate Achieved', 60, 220, 150)
    exercise_angina = st.selectbox('Exercise Induced Angina', ['N', 'Y'])
    oldpeak = st.number_input('ST Depression Induced by Exercise', 0.0, 10.0, 1.0, 0.1)
    st_slope = st.selectbox('Slope of Peak Exercise ST Segment', ['Down', 'Flat', 'Up'])
    num_major_vessels = st.selectbox('Number of Major Vessels (0-3)', [0, 1, 2, 3])
    thal = st.selectbox('Thalassemia', ['Fixed Defect', 'Normal', 'Reversible Defect'])

    # Create input dict with one-hot encoding like training
    input_dict = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': chol,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'NumMajorVessels': num_major_vessels
    }

    # Categorical one-hot columns (initialize 0)
    for col in feature_columns:
        if col not in input_dict:
            input_dict[col] = 0

    # Assign one-hot values based on user input
    input_dict[f'Sex_{sex}'] = 1
    input_dict[f'ChestPainType_{cp}'] = 1
    input_dict[f'FastingBS_{fasting_bs}'] = 1
    input_dict[f'RestingECG_{restecg}'] = 1
    input_dict[f'ExerciseAngina_{exercise_angina}'] = 1
    input_dict[f'ST_Slope_{st_slope}'] = 1
    # For thalassemia, note spaces replaced by underscores:
    thal_col = f'Thal_{thal.replace(" ", "_")}'
    input_dict[thal_col] = 1

    # Make DataFrame with columns in exact same order as training
    features = pd.DataFrame(input_dict, index=[0])[feature_columns]

    return features

def main():
    st.title('Heart Disease Prediction App')

    input_df = user_input_features()

    if st.button('Predict'):
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)

        st.write(f'Prediction: {"Heart Disease" if prediction[0]==1 else "No Heart Disease"}')
        st.write(f'Probability: {proba[0][1]:.2f}')

if __name__ == '__main__':
    main()
