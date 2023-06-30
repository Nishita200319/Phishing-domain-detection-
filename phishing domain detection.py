#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# In[2]:


# Load and analyze the dataset
data = pd.read_csv('dataset_full.csv')
print(data.head())


# In[3]:


# Summary statistics
print(data.describe())


# In[4]:


# Check the data types of each column
print(data.dtypes)


# In[5]:


data.isnull().any().any()


# In[6]:


# Split the dataset into input features and target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# In[7]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# Reshape the input features for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[9]:


# Create the RNN model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1, activation='sigmoid'))


# In[10]:


# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[12]:


# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)


# In[14]:


# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)


# In[ ]:




