import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Student Performance Prediction", layout="centered")

st.title("🎓 Student Final Grade (G3) Predictor")

# Load model
model = joblib.load("model.pkl")

st.markdown("Enter the student details below. The model will predict the final grade (G3, 0-20).")

with st.form("input_form"):
    # Numeric inputs
    age = st.number_input("Age", min_value=10, max_value=25, value=16)
    Medu = st.selectbox("Mother's education (0-4)", [0,1,2,3,4], index=2)
    Fedu = st.selectbox("Father's education (0-4)", [0,1,2,3,4], index=2)
    traveltime = st.selectbox("Home to school travel time (1:<15min, 2:15-30, 3:30-60, 4:>60)", [1,2,3,4], index=0)
    studytime = st.selectbox("Weekly study time (1:<2hrs, 2:2-5, 3:5-10, 4:>10)", [1,2,3,4], index=1)
    failures = st.number_input("Number of past class failures", min_value=0, max_value=10, value=0)
    famrel = st.selectbox("Family relationship quality (1-5)", [1,2,3,4,5], index=3)
    freetime = st.selectbox("Free time after school (1-5)", [1,2,3,4,5], index=2)
    goout = st.selectbox("Going out with friends (1-5)", [1,2,3,4,5], index=2)
    Dalc = st.selectbox("Workday alcohol consumption (1-5)", [1,2,3,4,5], index=1)
    Walc = st.selectbox("Weekend alcohol consumption (1-5)", [1,2,3,4,5], index=1)
    health = st.selectbox("Current health status (1-5)", [1,2,3,4,5], index=2)
    absences = st.number_input("Number of school absences", min_value=0, max_value=100, value=3)
    G1 = st.number_input("First period grade (G1, 0-20)", min_value=0, max_value=20, value=10)
    G2 = st.number_input("Second period grade (G2, 0-20)", min_value=0, max_value=20, value=10)

    # Categorical inputs - options populated from training data
    school = st.selectbox("School", ['GP', 'MS'])
    sex = st.selectbox("Sex", ['F', 'M'])
    address = st.selectbox("Address", ['U', 'R'])
    famsize = st.selectbox("Family size", ['GT3', 'LE3'])
    Pstatus = st.selectbox("Parent cohabitation status", ['A', 'T'])
    Mjob = st.selectbox("Mother's job", ['at_home', 'health', 'other', 'services', 'teacher'])
    Fjob = st.selectbox("Father's job", ['teacher', 'other', 'services', 'health', 'at_home'])
    reason = st.selectbox("Reason to choose this school", ['course', 'other', 'home', 'reputation'])
    guardian = st.selectbox("Guardian", ['mother', 'father', 'other'])
    schoolsup = st.selectbox("Extra educational support", ['yes', 'no'])
    famsup = st.selectbox("Family educational support", ['no', 'yes'])
    paid = st.selectbox("Extra paid classes", ['no', 'yes'])
    activities = st.selectbox("Extra-curricular activities", ['no', 'yes'])
    nursery = st.selectbox("Attended nursery", ['yes', 'no'])
    higher = st.selectbox("Wants higher education", ['yes', 'no'])
    internet = st.selectbox("Internet access at home", ['no', 'yes'])
    romantic = st.selectbox("In a romantic relationship", ['no', 'yes'])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build DataFrame in same column order and data types used for training
    data = [[
        age, Medu, Fedu, traveltime, studytime, failures, famrel,
        freetime, goout, Dalc, Walc, health, absences, G1, G2,
        school, sex, address, famsize, Pstatus, Mjob, Fjob, reason,
        guardian, schoolsup, famsup, paid, activities, nursery,
        higher, internet, romantic
    ]]

    columns = [
        'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
        'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
        'absences', 'G1', 'G2', 'school', 'sex', 'address', 'famsize',
        'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
        'famsup', 'paid', 'activities', 'nursery', 'higher',
        'internet', 'romantic'
    ]

    input_df = pd.DataFrame(data, columns=columns)

    pred = model.predict(input_df)[0]
    st.metric(label="Predicted final grade (G3 out of 20)", value=f"{pred:.2f}")
    st.write("Note: This prediction is approximate and model-dependent.")

# Footer
st.markdown('---')
st.write('Model: RandomForestRegressor pipeline (preprocessing + model).')
