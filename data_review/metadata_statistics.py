# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:07:06 2020

@author: Camila
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

my_path = os.path.abspath('C:/Users/Camila/Documents/Tesis/Proyecto/PlantillaTesis/Figures') # Figures out the absolute path for you in case your working directory moves around.

df = pd.read_csv('C:/Users/Camila/Documents/Tesis/raw_data/metadata.csv')
df = df.replace('m','Masculino')
df = df.replace('w','Femenino')
df = df.rename(columns={"gender": "Sexo", "age": "Edad"})

sns.set()
sns.set_palette("husl")
ax = sns.countplot(x="Sexo", data=df)
ax.set(xlabel='Sexo', ylabel='Conteo')
plt.legend('ABCDEF', ncol=2, loc='upper left');
ax.figure.savefig(os.path.join(my_path,'age.eps'))


# Distribución de edades por sexo

