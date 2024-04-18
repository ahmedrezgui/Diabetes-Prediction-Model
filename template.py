import numpy as np
import pickle
import streamlit as st



loaded_model=pickle.load(open('./diabetes-prediction-model.sav','rb'))

def predict_diabetes(input_data):
    input_data_as_array = np.array(input_data)
    input_data_reshaped=input_data_as_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    if(prediction[0]==0):
        return 'the person is not diabetic'
    else:
        return 'the person is diabetic' 
    

def main():
    st.title('Diabetes Prediction App')
    Pregnancies = st.text_input('Pregnancies')
    Glucose = st.text_input('Glucose')
    BloodPressure = st.text_input('BloodPressure')
    SkinThickness = st.text_input('SkinThickness')
    Insulin = st.text_input('Insulin')
    Bmi = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age')
    

    result = ''

    if st.button('Test Result'):
        result = predict_diabetes([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, Bmi, DiabetesPedigreeFunction, Age])
    
    st.success(result)    

if __name__=='__main__':
    main()    