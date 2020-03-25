"""
Created on Sun Mar 31 22:22:57 2019
@author: Camila
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline

dataframe = pd.read_csv("C:/Users/Camila/Documents/TESIS (We will make it)/AnalisisBF/data.csv")

 	
print(dataframe.groupby('clase').size())

dataframe.drop(['clase'],1).hist()
plt.show()

sb.pairplot(dataframe.dropna(), hue='clase',size=4,vars=["b_delta", "b_theta","b_alpha","b_beta","v_delta", "v_theta","v_alpha","v_beta"],kind='reg')

X = np.array(dataframe.drop(['canal','clase'],1))
y = np.array(dataframe['clase'])
X.shape

ep = dataframe.loc[y == 1]
noep = dataframe.loc[y == 0]

plt.scatter(ep.iloc[:, 0], ep.iloc[:, 1], s=10, label='Epilepticos')
plt.scatter(noep.iloc[:, 0], noep.iloc[:, 1], s=10, label='No Epilepticos')
plt.legend()
plt.show()

# Modelo de Regresion Logistica



model = linear_model.LogisticRegression()
model.fit(X,y)

predictions = model.predict(X)

model.score(X,y)