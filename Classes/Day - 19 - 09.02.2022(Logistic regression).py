#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# In[3]:


data=pd.read_csv('HR_comma_sep.csv')


# In[4]:


data.head()


# In[8]:


#left target varaible.
left=data[data.left==1]
left.shape


# In[7]:


retain=data[data.left==0]
retain.shape


# In[9]:


data.groupby('left').mean()


# In[11]:


pd.crosstab(data.salary,data.left).plot(kind='bar')


# In[12]:


pd.crosstab(data.Department,data.left).plot(kind='bar')


# In[13]:


sel_data = data[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
sel_data.head()


# In[14]:


dummies = pd.get_dummies(sel_data.salary)


# In[15]:


dummies


# In[16]:


data_dummies = pd.concat([sel_data,dummies],axis='columns')


# In[17]:


data_dummies


# In[18]:


data_dummies.drop('salary',axis='columns',inplace=True)
data_dummies.head()


# In[19]:


X = data_dummies
X.head()


# In[20]:


y=data.left


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)


# In[22]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[23]:


model.fit(X_train,y_train)


# In[29]:


y_pred=model.predict(X_test).reshape(-1,1)


# In[32]:


model.score(X_test,y_test)


# In[ ]:




