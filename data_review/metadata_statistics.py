# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:07:06 2020

@author: Camila
"""

# imports

import pandas as pd
import matplotlib.pyplot as plt


# read csv
df = pd.read_csv('C:/Users/Camila/Documents/Tesis/raw_data/metadata.csv')

df['age'].mean()
df['age'].mode()
df['age'].std()

## AGE
age = pd.qcut(df['age'],5, precision=0)
ax = age.value_counts(sort=False).plot.bar(rot=0, color="yellowgreen", figsize=(6,4))
ax.set_xticks(
        [r for i in range(len(age.cat.categories))],
        [str(age[c].left)+' a '+str(age[c].right) for c in age.cat.categories],
        rotate=90)
plt.show()

barWidth = 0.8
plt.bar(1, women, width = barWidth, color = (0.3,0.1,0.4,0.6), label='Mujeres')
plt.bar(2, men, width = barWidth, color = (0.3,0.5,0.4,0.6), label='Hombres')
plt.bar(3, n0, width = barWidth, color = (0.3,0.9,0.4,0.6), label='Sin info.')

# Create legend
plt.legend()
 
# Text below each barplot with a rotation at 90°
plt.xticks([r + barWidth for r in range(3)], ['Mujeres', 'Hombres', 'Sin info'], rotation=90)
 
# Create labels
label = ['n ='+str(women), 'n ='+str(men), 'n ='+str(n0)]
 
# Text on the top of each barplot
plt.text(x = 1 , y = women+0.1, s = label[0], size = 6)
plt.text(x = 2 , y = men+0.1, s = label[1], size = 6)
plt.text(x = 3 , y = n0+0.1, s = label[2], size = 6)
# Adjust the margins
plt.subplots_adjust(bottom= 0.2, top = 0.98)
 
# Show graphic
plt.show()





## SEX
women = df['gender'].value_counts()['w']
men = df['gender'].value_counts()['m']
n0 = df['gender'].isna().sum()

df['gender'].unique

barWidth = 0.8
plt.bar(1, women, width = barWidth, color = (0.3,0.1,0.4,0.6), label='Mujeres')
plt.bar(2, men, width = barWidth, color = (0.3,0.5,0.4,0.6), label='Hombres')
plt.bar(3, n0, width = barWidth, color = (0.3,0.9,0.4,0.6), label='Sin info.')

# Create legend
plt.legend()
 
# Text below each barplot with a rotation at 90°
plt.xticks([r + barWidth for r in range(3)], ['Mujeres', 'Hombres', 'Sin info'], rotation=90)
 
# Create labels
label = ['n ='+str(women), 'n ='+str(men), 'n ='+str(n0)]
 
# Text on the top of each barplot
plt.text(x = 1 , y = women+0.1, s = label[0], size = 6)
plt.text(x = 2 , y = men+0.1, s = label[1], size = 6)
plt.text(x = 3 , y = n0+0.1, s = label[2], size = 6)
# Adjust the margins
plt.subplots_adjust(bottom= 0.2, top = 0.98)
 
# Show graphic
plt.show()
