# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 22:45:19 2020

@author: Camila
"""


import numpy as np
import pandas as pd
import seaborn as sb
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
url = 'C:/Users/Camila/Documents/Tesis/csv/relative/data.csv'
eeg_dataset = pd.read_csv(url,error_bad_lines=False)
eeg_dataset.head()


X = eeg_dataset[['alpha','betha','delta','gamma','theta']].values
y = eeg_dataset[['class']].values.ravel()

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7,test_size = .3, random_state=25)

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler() 

# Don't cheat - fit only on training data
scaler.fit(X_train)  
X_train = scaler.transform(X_train) 

# apply same transformation to test data
X_test = scaler.transform(X_test) 

# =============================================================================
# clf.fit(X_train, y_train)
# 
# y_pred = clf.predict(X_train)
# 
# 
# confusion_matrix = confusion_matrix(y_train, y_pred)
# print(confusion_matrix)
# print(classification_report(y_train, y_pred))
# print(clf.score(X_train,y_train))
# =============================================================================


clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))
print(clf.score(X_test,y_test))
