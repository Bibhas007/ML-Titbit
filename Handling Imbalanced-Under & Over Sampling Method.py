#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn 
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams

rcParams['figure.figsize']=14,8
RANDOM_SEED=42
LABELS=["Normal","Fraud"]


# In[2]:


data = pd.read_excel(r'C:\Users\admin\Desktop\Course\DataSet\CreditCard.xlsx')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


columns=data.columns.tolist()
columns=[c for c in columns if c not in ["Class"]]
target="Class"
state=np.random.RandomState(42)
X=data[columns]
Y=data[target]

print(X.shape)
print(Y.shape)


# In[6]:


data=data.dropna(axis="rows")


# In[7]:


data.isnull().sum()


# In[8]:


columns=data.columns.tolist()
columns=[c for c in columns if c not in ["Class"]]
target="Class"
state=np.random.RandomState(42)
X=data[columns]
Y=data[target]

print(X.shape)
print(Y.shape)


# In[9]:


X.head()


# In[10]:


Y.head()


# In[11]:


count_classes=pd.value_counts(data['Class'],sort=True)
count_classes.plot(kind='bar',rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2),LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency");


# In[12]:


fraud = data[data['Class']==1]
normal = data[data['Class']==0]


# In[13]:


print(fraud.shape,normal.shape)


# In[14]:


from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss


# In[15]:


smk = SMOTETomek(random_state=42)
X_res,Y_res=smk.fit_sample(X,Y)


# In[16]:


print(X_res.shape,Y_res.shape)


# In[20]:


from collections import Counter
print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(Y_res)))


# In[21]:


from imblearn.over_sampling import RandomOverSampler


# In[24]:


os= RandomOverSampler(0.5)


# In[25]:


X_train_res,y_train_res=os.fit_sample(X,Y)


# In[26]:


X_train_res.shape,y_train_res.shape


# In[28]:


print("Original dataset shape {}".format(Counter(Y)))
print("Resampled dataset shape {}".format(Counter(y_train_res)))


# In[29]:


os_us=SMOTETomek(0.5)

X_train_res1,y_train_res1 = os_us.fit_sample(X,Y)


# In[30]:


X_train_res1.shape,y_train_res1.shape


# In[31]:


print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(y_train_res1)))


# In[ ]:




