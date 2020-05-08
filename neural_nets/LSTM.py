# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:46:43 2020

@author: Camila
"""


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

# Definiendo los datos de entrada y las clases

url = 'C:/Users/Camila/Documents/Tesis/csv/relative/data.csv'
eeg_dataset = pd.read_csv(url,error_bad_lines=False)
eeg_dataset.head()


X = eeg_dataset[['alpha','betha','delta','gamma','theta']].values
y = eeg_dataset[['class']].values.ravel()


# Segmentar los datos

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=.7,test_size = .3, random_state=25)


# Escalado de caracteristicas
from sklearn.preprocessing import RobustScaler  
scaler = RobustScaler() 

scaler.fit(X_train)  
x_train = scaler.transform(x_train) 

x_test = scaler.transform(x_test) 

# Arquitectura de modelo
max_features = 512

model = Sequential()
model.add(Embedding(max_features, output_dim=64))
model.add(LSTM(64))
model.add(Dropout(0.8))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=30)
score = model.evaluate(x_test, y_test, batch_size=16)
print(score)