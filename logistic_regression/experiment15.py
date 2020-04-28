# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:42:48 2020

@author: Camila
"""


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

rcParams['figure.figsize'] = 10, 8
sb.set_style('whitegrid')

url = "C:/Users/Camila/Documents/Tesis/csv/relative/data.csv"

eeg_dataset = pd.read_csv(url,error_bad_lines=False)
eeg_dataset.head()

X = eeg_dataset.iloc[0:105,3:8]
X = X.append(eeg_dataset.iloc[420:525,3:8])
X = X.values


y = eeg_dataset.iloc[0:105,2]
y = y.append(eeg_dataset.iloc[420:525,2])
y = y.values


#Split Training Set and Testing Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(X,y,test_size=0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
x_train=sc_X.fit_transform(x_train)
x_test=sc_X.transform(x_test)

#Training the Logistic Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=1,penalty='l2',solver='lbfgs')
classifier.fit(x_train, y_train)

#Predicting the Test Set Result
y_pred = classifier.predict(x_test)

#Create Confusion Matrix for Evaluation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
classifier.score(x_train,y_train)

print(cm)
print(classifier.score(x_train,y_train))
print(classification_report(y_test, y_pred))

columnas = ['No epiléptico','Epiléptico']

df_cm = pd.DataFrame(cm,index=columnas, columns=columnas)

grafica = sns.heatmap(df_cm,cmap=sns.color_palette("cubehelix", 8),annot=True,fmt='g')
plt.ylabel('Valores verdaderos')
plt.xlabel('Predicciones')
grafica.set(xlabel='Verdaderos',ylabel='Predicciones')
plt.show()