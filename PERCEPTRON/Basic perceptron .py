#Importing libraries 
import pandas as pd
import numpy as np

#Importing graph-plotting libraries 
import matplotlib.pyplot as plt
import seaborn as sns

#Loading Datasets 
df = pd.read_csv('placement.csv')

#Viewing top 5 rows of dataset
df.head()

#shape of dataset (no. of columns and rows)
df.shape

#plotting scatterplot 
sns.scatterplot(x= df['cgpa'] , y=df['resume_score'] , hue = df['placed'])

#NOTE : Here we had plotted scatter plot to just roughly check how the data is classified before applying perceptron.

#Splitting dataset into dependent variables and independent variables
X = df.iloc[: , 0:2]
y=df.iloc[: ,-1]

# Importing perceptron from sklearn library 
from sklearn.linear_model import Perceptron


# storing the perceptron object into variable p
p = Perceptron()
p

# fitting the X,y into perceptron object
p.fit(X,y)

# coef_ attribute is basically used to calculate the value of w1, w2  
p.coef_

#Conclusion : Here w1 = 40.26 , w2 = -36. 

#intercept_ is used to calculate the value of bias b
p.intercept_

#Conclusion : b = -25.

#installing mlxtend 
get_ipython().system('pip install mlxtend')

# Importing plot decision library which will split data into _ve and +ve regions 
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X.values, y.values ,clf =p  ,legend=2)

#Thank you !






