import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import sklearn as sklearn
import pickle
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(layout="wide")
image = Image.open("image.png")

st.image(image,width=1240)

st.markdown("<h1 style='text-align: center; color: black;'>Employee Churn Analysis</h1>", unsafe_allow_html=True)


#st.title('Employee Churn Analysis')
st.info('**We can make predictions according to the entered criterias with different machine learning models whether our employer will stay in our company or left our company.**')

st.markdown("""
<style>
.big-font {
    font-size:100px !important;
}
</style>
""", unsafe_allow_html=True)

st.subheader('Please select the maschine learning model')

mlmodel = st.selectbox(' ',['Gradient Boosting','KNN','RandomForest','XGBoost'])



st.sidebar.markdown('**Criterias for Employee Churn Analysis**')

satisfaction_level = st.sidebar.number_input('Satisfaction Level of Employer:',0.09,1.0,0.5,0.01)
last_evaluation = st.sidebar.number_input('Last Evaluation of Employer:',0.36,1.0,0.5,0.01)
number_project = st.sidebar.number_input('Number of Projects:',1,8,1,1)
average_montly_hours = st.sidebar.number_input('Average Monthly Hours:',90,320,100,1)
time_spend_company = st.sidebar.slider(' Time spend company(Workyear)', min_value=1, max_value=11, value=5, step=1) 
work_accident = st.sidebar.radio('Work Accident',  ["Yes", "No"])
promotion_last_5years = st.sidebar.radio('Promotion in last 5 years',  ["Yes", "No"])
departments = st.sidebar.selectbox('Select Department',  ['sales', 'accounting', 'hr', 'technical', 'support', 'management',
       'IT', 'product_mng', 'marketing', 'RandD'])
salary= st.sidebar.radio('Select Salary Level',  ['Low', 'Medium', 'High'])


my_dict = {
    "satisfaction_level": satisfaction_level,
    "last_evaluation": last_evaluation,
    "number_project": number_project,
    "average_montly_hours": average_montly_hours,
    "time_spend_company": time_spend_company,
    "Work_accident": work_accident,
    "promotion_last_5years": promotion_last_5years,
    "Departments": departments,
    "salary": salary,
    }

my_dict = pd.DataFrame([my_dict])


columns=[
 'satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'salary', 'Departments_IT',
       'Departments_RandD', 'Departments_accounting', 'Departments_hr',
       'Departments_management', 'Departments_marketing',
       'Departments_product_mng', 'Departments_sales', 'Departments_support',
       'Departments_technical']

codes = {'Low':0, 'Medium':1, 'High':2}
codes1 = {'Yes':1, 'No':0}

my_dict['salary'] = my_dict['salary'].map(codes)
my_dict['Work_accident'] = my_dict['Work_accident'].map(codes1)
my_dict['promotion_last_5years'] = my_dict['promotion_last_5years'].map(codes1)

final_encoder = pickle.load(open('encoder.pkl', 'rb'))
my_dict_cat = my_dict[['Departments']]

my_dict_num = my_dict.drop('Departments', axis=1)

my_dict_cat_enc = final_encoder.transform(my_dict_cat)

cols = final_encoder.get_feature_names(['Departments'])

my_dict_enc = my_dict_num.join(pd.DataFrame(my_dict_cat_enc, my_dict_num.index, columns=cols))

final_scaler = pickle.load(open('scaler.pkl', "rb"))

my_dict_scale = final_scaler.transform(my_dict_enc)

my_dict_scaled = pd.DataFrame(my_dict_scale, columns = my_dict_enc.columns)

if mlmodel == 'XGBoost':
    filename1 = "xgb.pkl"
    model = pickle.load(open(filename1, "rb"))
    pred = model.predict(my_dict_scale)
elif mlmodel == 'RandomForest':
    filename2 = 'rf.pkl'
    model = pickle.load(open(filename2, "rb"))
    pred = model.predict(my_dict_scaled) 
elif mlmodel == 'KNN':
    filename3 = 'knn.pkl'
    model = pickle.load(open(filename3, "rb"))
    pred = model.predict(my_dict_scaled)
else:
    filename4 = "gradboosting.pkl"
    model = pickle.load(open(filename4, "rb"))
    pred = model.predict(my_dict_scaled)

st.write(' ')

if st.button('Predict'):
    if int(pred[0]) == 0:
        st.success('This Employer will Stay')
        image_stay = Image.open("employee_stay.png")
        st.image(image_stay,width=90)
        #st.balloons()
    elif int(pred[0]) == 1:
        st.error('This Employer will Left')
        image_left = Image.open("employee_left.png")
        st.image(image_left,width=100)
        #st.snow() 








