#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Importing libraries 
import pandas as pd
import numpy as np

# Graph plotting libraries 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Loading Datasets 
df = pd.read_csv('placement.csv')


# In[3]:


#Viewing top 5 rows of dataset
df.head()


# In[4]:


#shapeof dataset (no. of columns and rows)
df.shape


# In[7]:


#plotting scatterplot 
sns.scatterplot(x= df['cgpa'] , y=df['resume_score'] , hue = df['placed'])


# In[8]:


#Splitting dataset into dependent variables and independent variables

X = df.iloc[: , 0:2]
y=df.iloc[: ,-1]


# In[10]:


# Importing perceptron from sklearn library 
from sklearn.linear_model import Perceptron


# In[12]:


# storing the perceptron object into variable p
p = Perceptron()
p


# In[13]:


# fitting the X,y into perceptron object
p.fit(X,y)


# In[14]:


# coef_ attribute is basically used to calculate the value of w1, w2  
p.coef_

#Conclusion : Here w1 = 40.26 , w2 = -36. 


# In[15]:


#intercept_ is used to calculate the value of bias b
p.intercept_

#Conclusion : b = -25.


# In[18]:


#installing mlxtend 
get_ipython().system('pip install mlxtend')


# In[19]:


# Importing plot decision library which will split data into _ve and +ve regions 
from mlxtend.plotting import plot_decision_regions


# In[20]:


plot_decision_regions(X.values, y.values ,clf =p  ,legend=2)

Thank you !

# In[ ]:




