#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.datasets import load_iris


# In[3]:


from sklearn.model_selection import train_test_split


# In[4]:


from sklearn.tree import DecisionTreeClassifier


# In[5]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


iris = load_iris()
X = iris.data
y = iris.target


# In[8]:


df = pd.DataFrame(data=X, columns=iris.feature_names)
df['target'] = y


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# In[11]:


y_pred = model.predict(X_test)


# In[12]:


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[13]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




