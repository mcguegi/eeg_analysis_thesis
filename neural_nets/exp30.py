# -*- coding: utf-8 -*-
"""
Created on Mon May 11 23:01:07 2020

@author: Camila
"""


# -*- coding: utf-8 -*-
"""
Created on Sat May  2 11:15:01 2020

@author: Camila
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Lambda
import pandas as pd

# Definiendo los datos de entrada y las clases

url = 'C:/Users/Camila/Documents/Tesis/csv/relative/data.csv'
eeg_dataset = pd.read_csv(url,error_bad_lines=False)
eeg_dataset.head()


X = eeg_dataset[['alpha','betha','delta','gamma','theta']].values
y = eeg_dataset[['class']].values.ravel()


# Segmentar los datos

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7,test_size = .3, random_state=25)

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler() 

# Don't cheat - fit only on training data
scaler.fit(X_train)  
X_train = scaler.transform(X_train) 

# apply same transformation to test data
X_test = scaler.transform(X_test) 

# Definir el modelo Secuencial

model = Sequential()
model.add(Dense(128, input_dim=5, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Lambda(lambda x: x ** 4))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo

model.compile(loss='mean_absolute_error',
              optimizer='nadam',
              metrics=['accuracy'])


# Entrenar el modelo

model.fit(X_train, y_train,
          epochs=1000,
          batch_size=32)

score = model.evaluate(X_test, y_test, batch_size=128)
print(score)

import matplotlib.pyplot as plt

history = model.fit(X, y, validation_split=0.3, epochs=50, batch_size=128, verbose=1)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Precisión del modelo')
plt.ylabel('Precisión')
plt.xlabel('Epoch')
plt.legend(['Entrenamiento', 'Prueba'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Pérdida del modelo')
plt.ylabel('Pérdida')
plt.xlabel('Epoch')
plt.legend(['Entrenamiento', 'Prueba'], loc='upper left')
plt.show()


score = model.evaluate(X_test, y_test, batch_size=128)

from ann_visualizer.visualize import ann_viz;
import os.path
from keras.utils import plot_model

save_path = 'C:/Users/Camila/Documents/Tesis/'
plot_model(model, to_file=os.path.join(save_path,"NN-FF01.png"))


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'



ann_viz(model, view=True, filename=os.path.join(save_path,"NN-FF01.gv"), title="NN-FF01")

