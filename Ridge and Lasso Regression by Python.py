#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=load_boston()


# In[3]:


df


# In[4]:


DataSet=pd.DataFrame(df.data,columns=df.feature_names)


# In[5]:


DataSet.head(2)


# In[6]:


DataSet["Price"]=df.target


# In[7]:


DataSet.head(2)


# In[8]:


X=DataSet.iloc[:,:-1]## Independent features
y=DataSet.iloc[:,-1]## Dependent features


# ## Linear Regression

# In[9]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


# In[10]:


Linear_Regression=LinearRegression()
Mean_Squared_Error = cross_val_score(Linear_Regression,X,y,scoring='neg_mean_squared_error',cv=5)


# In[11]:


Mean_Squared_Error


# In[12]:


Mean_Mean_Squared_Error = np.mean(Mean_Squared_Error)
Mean_Mean_Squared_Error


# ## Ridge Regression

# In[13]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge


# In[14]:


ridge=Ridge()
#ridge = Ridge()
Parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
Ridg_Regression = GridSearchCV(ridge,Parameters,scoring='neg_mean_squared_error',cv=5)
Ridg_Regression.fit(X,y)


# In[15]:


print(Ridg_Regression.best_params_)


# In[16]:


print(Ridg_Regression.best_score_)


# ## Lasso Regression

# In[17]:


from sklearn.linear_model import Lasso


# In[18]:


lasso=Lasso()
lasso_regression=GridSearchCV(lasso,Parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regression.fit(X,y)


# In[19]:


print(lasso_regression.best_params_)


# In[20]:


print(lasso_regression.best_score_)


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[30]:


prediction_lasso=lasso_regression.predict(X_test)
prediction_ridge=Ridg_Regression.predict(X_test)


# ## Visualization of Ridge & Lasso

# In[50]:


import seaborn as sns
plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
sns.distplot(y_test-prediction_lasso)
plt.title('Lasso Regression & Ridge Regression')
plt.subplot(2,1,2)
sns.distplot(y_test-prediction_ridge)
plt.show()


# In[ ]:




