#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 16:29:25 2023

@author: thear
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import zscore

# 117 s of data
# 0 is eyes-open
# 1 is eyes-closed

# Import data from CSV to dataframe
data = pd.read_csv('EEG_Eye_State_Classification.csv')

# Inspect data
print(data.head())
print(data.info())
data_desc = data.describe()
print(data_desc)
n_eye_open = data['eyeDetection'].value_counts()[0]
n_eye_closed = data['eyeDetection'].value_counts()[1]
print(f'Percent Eyes Open: {100*n_eye_open/(n_eye_open+n_eye_closed):.2f}')

# Create time vector
t = np.linspace(0, 117, num = data.shape[0])

# Visualize EEG data 
for col in data.columns[0:-1]:
    plt.figure(figsize = (12, 3))
    sns.lineplot(x = t, 
                  y = col,
                  data = data,
                  hue = "eyeDetection")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (uV)')
    plt.title(col)
    plt.show()

# Visualize eyeDetection
plt.figure(figsize = (12, 3))
sns.lineplot(x = t,
             y = data['eyeDetection'])
plt.show()

# Plot correlation matrix
corr_mat = data.corr()
plt.figure(figsize = (15, 15))
sns.heatmap(corr_mat,
            annot = True,
            cmap = 'magma')

# Large voltage spikes present, need to clean data
# create sample data

# Determine outliers based on z-score
z_scores = zscore(data)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 10).all(axis=1)
data_filt = data[filtered_entries]
# reset index
data_filt = data_filt.reset_index(drop=True)
# Display Features after outliers removed
for f in data_filt.columns:
    data_filt[f].plot(figsize = (12, 3),
                         title = f)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (uV)')
    plt.show()


# Split  data into features/labels
X = data_filt.copy()
y = X.pop('eyeDetection')

# Helpful variables
n_chan = X.shape[1]
n_samp = X.shape[0]

# Create the scaler object
scaler = StandardScaler()

# Fit the scaler to the selected columns
X_scaled = scaler.fit_transform(X)

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size = 0.2,
    random_state = 42,
    shuffle = True,
    stratify = None
    )





def design_model(X):
    # Initialize model
    model = Sequential()
    # Create input layer
    model.add(InputLayer(input_shape = (X.shape[1],1)))
    # Create LSTM layer
    model.add(LSTM(64, activation = 'relu'))
    # Create Dense layer
    model.add(Dense(32, activation = 'relu'))
    # Create Dense layer
    model.add(Dense(16, activation = 'relu'))
    # # Create Dense layer
    # model.add(Dense(8, activation = 'relu'))
    # Creat output layer 
    model.add(Dense(1, activation = 'sigmoid'))
    
    # Create optimizer
    opt = Adam(learning_rate = 0.001)
    # Compile model
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = opt,
        metrics = ['accuracy']
        )
    return model

# Apply the model
model = design_model(X_train)

# Add EarlyStopping for effiency
es = EarlyStopping(
    monitor = 'val_accuracy', 
    mode = 'min',
    verbose = 1,
    patience = 20
    )

# Fit the model
b_size = 20
n_epochs = 75
history = model.fit(X_train, 
                    y_train, 
                    batch_size = b_size,
                    epochs = n_epochs,
                    validation_split = 0.2,
                    verbose = 1,
                    callbacks = [es]
                    )

# Create model summary
model.summary()

# Evaluate the model
loss, acc = model.evaluate(
    X_test, 
    y_test, 
    verbose = 1)

# Make prediction
y_pred = model.predict(X_test)
# Convert prediction to screte values
y_pred = (y_pred > 0.5).astype(int)
class_names = ['Eyes Open', 'Eyes Closed']
# Print classification report
print(classification_report(
    y_test,
    y_pred, 
    target_names = class_names)
    )

# Plot Accuracy and Validation Accuracy over epochs
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(['Train', 'Validation'], loc='upper left')

# Plot Loss and Validation Loss over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(['Train', 'Validation'], loc='upper left')

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax3 = plt.subplots(figsize=(15, 15))
heatmap = sns.heatmap(
    cm, 
    fmt = 'g', 
    cmap = 'mako_r', 
    annot = True, 
    ax = ax3)
ax3.set_xlabel('Predicted class')
ax3.set_ylabel('True class')
ax3.set_title('Confusion Matrix')
ax3.xaxis.set_ticklabels(class_names)
ax3.yaxis.set_ticklabels(class_names)








