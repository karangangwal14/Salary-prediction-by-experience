import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import streamlit as st

salary = pd.read_csv("Salary_Data.csv")
model=smf.ols("Salary~YearsExperience",data=salary).fit()

st.title("Salary Prediction")
st.image("download.png",width=500)

st.header("Know your salary")
val=st.number_input("Enter your Experience",min_value=0.00,max_value=20.00,step=0.5)
prd=pd.Series([val])
dp=pd.DataFrame(prd,columns=['YearsExperience'])
predict=model.predict(dp)[0]

if st.button('Predict'):
   st.success(f"Your predicted salary is {round(predict)} ") 


