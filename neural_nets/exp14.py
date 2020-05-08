# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 21:18:22 2020

@author: Camila
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

# Definiendo los datos de entrada y las clases

url = 'C:/Users/Camila/Documents/Tesis/csv/relative/data.csv'
eeg_dataset = pd.read_csv(url,error_bad_lines=False)
eeg_dataset.head()


X = eeg_dataset[['alpha','betha','delta','gamma','theta']].values
y = eeg_dataset[['class']].values.ravel()


# Segmentar los datos

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7,test_size = .3, random_state=25)

# Definir el modelo Secuencial

model = Sequential()
model.add(Dense(64, input_dim=5, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# Entrenar el modelo

model.fit(X_train, y_train,
          epochs=20,
          batch_size=128)

score = model.evaluate(X_test, y_test, batch_size=128)