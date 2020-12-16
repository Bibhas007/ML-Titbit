#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = sns.load_dataset('iris')


# In[3]:


df.head()


# In[4]:


df['species'].unique()


# In[5]:


#Univariate Analysis
df_setosa = df.loc[df['species'] == 'setosa']
df_virginica = df.loc[df['species'] == 'virginica']
df_versicolor = df.loc[df['species'] == 'versicolor']


# In[6]:


plt.plot(df_setosa['sepal_length'],np.zeros_like(df_setosa['sepal_length']),'o')
plt.plot(df_virginica['sepal_length'],np.zeros_like(df_virginica['sepal_length']),'o')
plt.plot(df_versicolor['sepal_length'],np.zeros_like(df_versicolor['sepal_length']),'o')
plt.show()


# In[7]:


#Bivariate Analysis
plt.plot(df_setosa['sepal_length'],df_setosa['sepal_width'],'o')
plt.plot(df_virginica['sepal_length'],df_virginica['sepal_width'],'o')
plt.plot(df_versicolor['sepal_length'],df_versicolor['sepal_width'],'o')
plt.show()


# In[8]:


#Bivariate Analysis
sns.FacetGrid(df,hue='species',size=5).map(plt.scatter,'sepal_length','sepal_width').add_legend();
plt.show()


# In[9]:


#Univariate Analysis
sns.pairplot(df,hue='species')


# In[10]:


df=sns.load_dataset('tips')
df.head()


# In[11]:


#Cumulitive Density Function
plt.hist(df['total_bill'],cumulative=True,density=True,bins=10);


# In[17]:


kwargs = {'cumulative': True}
sns.distplot(df['tip'], hist_kws=kwargs, kde_kws=kwargs)


# In[27]:


import numpy as np
from matplotlib import pyplot as plt

#def my_dist(x):
#    return np.exp(-x ** 2)

#x = np.arange(-100, 100)
#p = my_dist(df['tip'])
plt.plot(df['tip'])
plt.show()


# In[ ]:




