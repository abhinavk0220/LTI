# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:28:10 2025

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:16:28 2024

@author: admin
"""
'''
conda create -n name_of_venv python=3.8
conda env list
conda activate name_of_venv
conda install tensorflow / pip install tensorflow
conda install keras/pip install keras
conda install spyder

pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import History


# Weights
# Bias
# Input Layers
# hidden Layers --> Neurons
# Forward Propagation
# Activation functions  : Sigmoid, RELU, SOFTMAX
# Loss fn/ cost fn
# Optimizers   : GD, SGD, Momentum, ADAGRAD, ADADELTA, ADAM, RMS Prop
# Back-propagation
# Weights adjustments (chain rule of differentiation)
 

# Loss functions : 

# MSE : sum(yi-e(y))2

# Binary Cross-entropy : -y*(log(y^)-(1-y)*log(1-y^))

# Multi-class cross entropy : -yij*log(yij^)

# Activation functions

# RELU : max(o,x)

from matplotlib import pyplot as plt
def relu_ex(x):
    return max(0.0,x)

inp = [i for i in range(-10,10)]
op = [relu_ex(i) for i in inp]
plt.plot(inp,op)
plt.show()

# Sigmoid : 1/1+exp(-z)

from math import exp

def sigmoid_ex(x):
    return 1.0/(1.0+exp(-x))
op_sig = [sigmoid_ex(i) for i in inp]
plt.plot(inp,op_sig)
plt.show()

# Tanh, Leaky Relu, Parameterised relu, exponential relu, 

# Softmax

def softmax_ex(x):
    y = np.exp(x) 
    y_ = y/y.sum()
    return y_

softmax_ex([0.8,1.2,3.1])
    
# Activation fns for hidden layers : RELU

# Activation fns for output layers

# 1. Binary classifiers : Sigmoid
# 2. Multi-class : Softmax
# 3. Regression  : RELU

# Optimizers :

# Gradient Descent : All the dataset is fed at a time
# Stochastic Gradient Descent : Each record is fed at a time
# Mini batch GD : Pass data by batches of records
# Momentum : Exponential weight average (taking previous learning rate by time steps into consideration)
# Adaptive learning rates : Taking learning rate into consideration
# ADAM : Combination Adaptive learning rates and momentum
# RMS Prop : Root mean squared error

df = pd.read_csv("E://data/churn.csv")

x = df.iloc[:,2:12]
y = df['Exited']

x1 = pd.get_dummies(x,columns=['Geography','Gender'], drop_first=True)
x1 = x1.values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x1,y,test_size=0.2, random_state = 2)

y_train = y_train.values
y_test = y_test.values

# Here we scale the data for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




# Build the ANN model
model = Sequential()

model.add(Dense(6, activation='relu',kernel_initializer='uniform',input_shape=(X_train_scaled.shape[1],)))  # First hidden layer
model.add(Dense(6, activation='relu',kernel_initializer = 'uniform'))  # Second hidden layer
model.add(Dense(1, activation='sigmoid', kernel_initializer = 'uniform'))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Precision'])

# Train the model and capture the training history
history = model.fit(X_train_scaled, y_train, epochs=500, batch_size=32, validation_split=0.2)

# Make predictions on the test set
predictions = model.predict(X_test_scaled)
predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions

pd.crosstab(y_test, predictions.reshape(1,-1))
X_test_scaled.shape

# Plot the training and validation loss
plt.figure(figsize=(12, 5))

# Plot for Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot for Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['precision'], label='Training precision')
plt.plot(history.history['val_precision'], label='Validation precision')
plt.title('precision over Epochs')
plt.xlabel('Epochs')
plt.ylabel('precision')
plt.legend()

plt.tight_layout()
plt.show()

# Visualize the weights of the first hidden layer
weights, biases = model.layers[0].get_weights()
plt.figure(figsize=(10, 6))
plt.imshow(weights, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Weights of First Hidden Layer')
plt.xlabel('Neurons')
plt.ylabel('Input Features')
plt.show()

print("Model training complete and results visualized.")



















