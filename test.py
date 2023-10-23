import numpy as np
import schedule
import pandas as pd
import json
import tensorflow as tf
import time
from urllib.request import urlopen
import warnings
warnings.filterwarnings("ignore") #suppress warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam

#getting actual values
airData = json.load(urlopen('https://api.data.gov.sg/v1/environment/air-temperature'))
humidData = json.load(urlopen('https://api.data.gov.sg/v1/environment/relative-humidity'))

airTempReadings = airData['items'][0]['readings']
humidReadings = humidData['items'][0]['readings']
#displaying results

for i in airTempReadings:
    if i['station_id'] == "S104" :
        aD = i['value']
        time.sleep(3.5)

print("\n")
print("Humidity")
for i in humidReadings:
    if i['station_id'] == 'S104' :
        hD = i['value']
        time.sleep(3.5)

# Getting actual values from JSON
airTempReadings = [float(reading['value']) for reading in airData['items'][0]['readings']]

# Splitting the data
Xtrain, Xtest, ytrain, ytest = train_test_split(airTempReadings, airTempReadings, test_size=0.2, random_state=2)

# Convert to NumPy arrays
Xtrain = np.array(Xtrain)
Xtest = np.array(Xtest)
ytrain = np.array(ytrain)
ytest = np.array(ytest)

# Convert to TensorFlow tensors
Xtrain = tf.convert_to_tensor(Xtrain)
Xtest = tf.convert_to_tensor(Xtest)
ytrain = tf.convert_to_tensor(ytrain)
ytest = tf.convert_to_tensor(ytest)

# Model for training (regression)
model = Sequential()
model.add(Dense(16, input_shape=(1,), activation='relu'))
model.add(Dense(8, input_shape=(1,), activation='relu'))
model.add(Dense(8, input_shape=(1,), activation='relu'))
model.add(Dense(1, activation='linear'))  # Using 'linear' activation for regression
opt = Adam(learning_rate=0.020)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])
model.fit(Xtrain, ytrain, epochs=100, verbose=1)

predictions = model.predict(airTempReadings)