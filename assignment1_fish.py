#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load and preprocess the data
data = pd.read_csv('Fish.csv')
# Preprocess and feature engineering steps go here

# Train-test split
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Fit ARIMA model
model = ARIMA(train_data['sales'], order=(p, d, q))
model_fit = model.fit()

# Make predictions for next year
future_data = pd.DataFrame()  # Create a dataframe with future dates and relevant features
forecast = model_fit.forecast(steps=len(future_data))

# Visualize predictions
plt.plot(train_data['timestamp'], train_data['sales'], label='Training Data')
plt.plot(test_data['timestamp'], test_data['sales'], label='Testing Data')
plt.plot(future_data['timestamp'], forecast, label='Predictions')
plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('Sales')
plt.show()


# In[3]:


pip install statsmodels


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load and preprocess the data
data = pd.read_csv('Fish.csv')
# Preprocess and feature engineering steps go here

# Train-test split
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Fit ARIMA model
model = ARIMA(train_data['sales'], order=(p, d, q))
model_fit = model.fit()

# Make predictions for next year
future_data = pd.DataFrame()  # Create a dataframe with future dates and relevant features
forecast = model_fit.forecast(steps=len(future_data))

# Visualize predictions
plt.plot(train_data['timestamp'], train_data['sales'], label='Training Data')
plt.plot(test_data['timestamp'], test_data['sales'], label='Testing Data')
plt.plot(future_data['timestamp'], forecast, label='Predictions')
plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('Sales')
plt.show()


# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('Fish.csv')

columns = ["Species", "Weight", "Length1", "Length2", "Length3", "Height", "Width"]

df = pd.DataFrame(data, columns=columns)

# Convert categorical species into numerical labels
df['Species'] = pd.Categorical(df['Species']).codes

# Prepare the data
X = df.drop("Weight", axis=1)  # Features
y = df["Weight"]  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[7]:


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


# In[21]:


# Select a Roach instance from the test set for prediction (you can adjust the index)
roach_instance = X_test.iloc[[0]]

# Transform features using the same polynomial transformation
roach_instance_poly = poly.transform(roach_instance)

# Predict weight for the Roach
roach_weight_prediction = model.predict(roach_instance_poly)

print("Predicted weight for Roach:", roach_weight_prediction)


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


# In[ ]:




