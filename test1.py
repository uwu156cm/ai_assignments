#!/usr/bin/env python
# coding: utf-8

# In[19]:


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


# In[20]:


# Convert species to numerical labels
df['Species'] = pd.Categorical(df['Species']).codes


# In[21]:


# Prepare data
X = df.drop("Weight", axis=1)
y = df["Weight"]


# In[22]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


# Create polynomial features
degree = 2  # Degree of the polynomial
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# In[24]:


# Train polynomial regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)


# In[25]:


# Make predictions
y_pred = model.predict(X_test_poly)


# In[26]:


# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[27]:


# Print evaluation results
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[31]:


# Create a dictionary to map numerical codes to species names
species_mapping = dict(zip(range(len(df['Species'].unique())), df['Species'].unique()))

# Streamlit app
st.title('Fish Market Assignment1')
st.write('Predicting the weight of different fish species.')

# Input form
st.header('Enter Fish Features for Prediction')
species_code = st.selectbox('Species', list(species_mapping.keys()))  # Dropdown for Species
species = species_mapping[species_code]  # Map code to species name

length1 = st.number_input('Length1', min_value=0.0, max_value=100.0, value=24.0)
length2 = st.number_input('Length2', min_value=0.0, max_value=100.0, value=25.0)
length3 = st.number_input('Length3', min_value=0.0, max_value=100.0, value=26.0)
height = st.number_input('Height', min_value=0.0, max_value=100.0, value=27.0)
width = st.number_input('Width', min_value=0.0, max_value=100.0, value=5.5)


# In[32]:


# Predict weight for the Fish species
fish_features = pd.DataFrame({
    "Species": [species],
    "Length1": [length1],
    "Length2": [length2],
    "Length3": [length3],
    "Height": [height],
    "Width": [width]
})

# Convert species to numerical label
fish_features['Species'] = pd.Categorical(fish_features['Species'], categories=df['Species'].unique()).codes

# Transform features using the same polynomial transformation
fish_features_poly = poly.transform(fish_features)

# Predict weight
fish_weight_prediction = model.predict(fish_features_poly)

st.write('Predicted weight for : ',fish_weight_prediction)


# In[30]:


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


# In[ ]:




