# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:52:09 2020

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns

rcParams['figure.figsize'] = 10, 8
sb.set_style('whitegrid')

url = 'C:/Users/Camila/Documents/Tesis/csv/relative/data.csv'
eeg_dataset = pd.read_csv(url,error_bad_lines=False)
eeg_dataset.head()

sb.countplot(x='class',data=eeg_dataset, palette='hls')


eeg_dataset.isnull().sum()

eeg_dataset.info()

sb.heatmap(eeg_dataset.corr())  

X = eeg_dataset[['alpha','betha','delta','gamma','theta']].values
y = eeg_dataset[['class']].values.ravel()

# Filtrando los sanos y los epilépticos
healthy = eeg_dataset.loc[y == 0]
unhealthy = eeg_dataset.loc[y == 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7,test_size = .3, random_state=25)

LogReg = LogisticRegression(C=1,random_state=20,solver='liblinear',penalty='l1')
LogReg.fit(X_train, y_train)

y_pred = LogReg.predict(X_train)


confusion_matrix = confusion_matrix(y_train, y_pred)
print(confusion_matrix)
print(classification_report(y_train, y_pred))
print(LogReg.score(X_train,y_train))

columnas = ['No epiléptico','Epiléptico']

df_cm = pd.DataFrame(confusion_matrix,index=columnas, columns=columnas)

grafica = sns.heatmap(df_cm,cmap=sns.color_palette("cubehelix", 8),annot=True,fmt='g')
plt.ylabel('Valores verdaderos')
plt.xlabel('Predicciones')
grafica.set(xlabel='Verdaderos',ylabel='Predicciones')
plt.show()

accuracy = (tp+tn)/(tp+tn+fp+fn)
recall = tp/(tp+fn)
precision = tp/(tp+fp)
f1 = (2*recall*precision)/(recall+precision)

from sklearn.metrics import precision,recall,f1_score

precision(y_test, y_pred)