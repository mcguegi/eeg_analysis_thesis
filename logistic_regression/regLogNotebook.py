
# coding: utf-8

# # Logistic Regression

# In[2]:


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


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 10, 8
sb.set_style('whitegrid')


# ## Logistic regression on the eeg dataset
# The first thing we are going to do is to read in the dataset using the Pandas' read_csv() function.
# In[3]:


url = 'C:/Users/Camila/Documents/Tesis/csv/dataFull.csv'
eeg_dataset = pd.read_csv(url,error_bad_lines=False, header=None)
eeg_dataset.head()

# In[4]:


sb.countplot(x='class',data=eeg_dataset, palette='hls')


# Ok, so we see that the class variable is binary (0 - non-epileptic / 1 - epileptic)
# 
# ### Checking for missing values
# It's easy to check for missing values by calling the isnull() method, and the sum() method off of that, to return a tally of all the True values that are returned by the isnull() method.

# In[5]:


eeg_dataset.isnull().sum()


# Well, how many records are there in the data frame anyway?

# In[6]:


eeg_dataset.info()

# In[18]:


sb.heatmap(eeg_dataset.corr())  

# In[21]:

X = eeg_dataset.iloc[1:,1:6].values
y = eeg_dataset[:,8].values
# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7,test_size = .3, random_state=25)


# ### Deploying and evaluating the model

# In[23]:


LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)


# In[24]:


y_pred = LogReg.predict(X_test)


# In[25]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix


# In[26]:

print(classification_report(y_test, y_pred))
