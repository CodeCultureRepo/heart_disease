import streamlit as st
import pandas as pd
import numpy as np
import joblib

joblib.dump(best_model, 'best_model.pkl')

model = joblib.load('best_model.pkl')

def get_user_input():
    age = st.number_input('Age', min_value=1, max_value=120, value=25)
    sex = st.selectbox('Sex', [0, 1])  # 0 for female, 1 for male
    cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=0, max_value=300, value=120)
    chol = st.number_input('Serum Cholesterol in mg/dl (chol)', min_value=0, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
    restecg = st.selectbox('Resting ECG (restecg)', [0, 1, 2])
    thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=0, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])
    oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment (slope)', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (ca)', [0, 1, 2, 3, 4])
    thal = st.selectbox('Thalassemia (thal)', [0, 1, 2, 3])
    
    user_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

def main():
    st.title('Heart Disease Prediction App')
    st.write('Enter the details of the patient to predict the likelihood of heart disease.')

    input_df = get_user_input()

    st.subheader('Patient Data')
    st.write(input_df)

    preprocessed_input = preprocessor.transform(input_df) 

    prediction = model.predict(preprocessed_input)
    prediction_proba = model.predict_proba(preprocessed_input)[:, 1]

    st.subheader('Prediction')
    heart_disease_risk = 'High' if prediction[0] == 1 else 'Low'
    st.write(f'The risk of heart disease is: {heart_disease_risk}')
    st.write(f'Probability of having heart disease: {prediction_proba[0]:.2f}')

if __name__ == '__main__':
    main()