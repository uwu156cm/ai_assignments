#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
st.title("BMI")
weight = st.number_input("Enter weight in kg")
height =st.number_input("Enter height in cm")
try:
    bmi =weight/((height/100)**2)
except:
    st.text("Enter value of height")
if(st.button("Calculate BMI")):
    st.text("Your BMI Index is {}.".format(bmi))
    if (bmi<16):
        st.error ("Extremely Underweight")
    elif (bmi>= 16 and bmi <18.5):
        st.warning ("Underweight")
    elif (bmi>= 18.5 and bmi <25):
        st.success ("Healthy")
    elif (bmi>= 25 and bmi <30):
        st.warning ("Overweight")
    elif (bmi >= 30):
        st.error ("Very Overweight")


# In[ ]:




