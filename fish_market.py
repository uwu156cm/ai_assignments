#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('Fish.csv')
columns = ["Species", "Weight", "Length1", "Length2", "Length3", "Height", "Width"]

df = pd.DataFrame(data, columns=columns)


# In[8]:


# Convert species to numerical labels
df['Species'] = pd.Categorical(df['Species']).codes


# In[9]:


# Prepare data
X = df.drop("Weight", axis=1)
y = df["Weight"]


# In[10]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


# Create polynomial features
degree = 2  # Degree of the polynomial
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# In[12]:


# Train polynomial regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)


# In[13]:


# Make predictions
y_pred = model.predict(X_test_poly)


# In[14]:


# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[15]:


# Print evaluation results
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[3]:


# Streamlit app
st.title('Fish Market Assignment1')
st.write('Predict the weight of a Roach species using a trained polynomial regression model.')

# Input form
st.header('Enter Roach Features for Prediction')
species = st.selectbox('Species', df['Species'].unique())  # Dropdown for Species
length1 = st.number_input('Length1', min_value=0.0, max_value=100.0, value=24.0)
length2 = st.number_input('Length2', min_value=0.0, max_value=100.0, value=25.0)
length3 = st.number_input('Length3', min_value=0.0, max_value=100.0, value=26.0)
height = st.number_input('Height', min_value=0.0, max_value=100.0, value=27.0)
width = st.number_input('Width', min_value=0.0, max_value=100.0, value=5.5)


# In[21]:


# Select a Roach instance from the test set for prediction (you can adjust the index)
roach_instance = X_test.iloc[[0]]

# Transform features using the same polynomial transformation
roach_instance_poly = poly.transform(roach_instance)

# Predict weight for the Roach
roach_weight_prediction = model.predict(roach_instance_poly)

print("Predicted weight for Roach:", roach_weight_prediction)


# In[22]:


#Streamlit app
# Predict weight for the Roach
roach_features = pd.DataFrame({
    "Species": [species],
    "Length1": [length1],
    "Length2": [length2],
    "Length3": [length3],
    "Height": [height],
    "Width": [width]
})

# Convert species to numerical label
roach_features['Species'] = pd.Categorical(roach_features['Species']).codes

# Transform features using the same polynomial transformation
roach_features_poly = poly.transform(roach_features)

# Predict weight
roach_weight_prediction = model.predict(roach_features_poly)

st.write('Predicted weight for Roach:', roach_weight_prediction)


# In[26]:


import matplotlib.pyplot as plt

# Plot scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Ideal')
plt.xlabel('Actual Weight')
plt.ylabel('Predicted Weight')
plt.title('Actual vs. Predicted Weight')
plt.legend()
plt.show()

