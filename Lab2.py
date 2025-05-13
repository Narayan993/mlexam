#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing


# In[3]:


data = fetch_california_housing()
df = pd.DataFrame(data.data , columns=data.feature_names)

corr_matrix = df.corr()

plt.figure(figsize=(10 , 8))
sns.heatmap(corr_matrix , annot=True ,  cmap="coolwarm" , fmt='.2f' , linewidths=0.5)
plt.title("Correleation heatmap for california housing dataset")
plt.show()


# In[4]:


sns.pairplot(df)
plt.show()


# Tips Dataset

# In[32]:


tips = sns.load_dataset("tips")
print(tips)


# In[31]:


sns.pairplot(tips)
plt.title("pairplot of tips dataset")
plt.show()


# In[25]:


sns.heatmap(tips.corr(numeric_only=True) , annot=True ,fmt='.2f', cmap='coolwarm' , linewidths=0.5)
plt.title("Correleation heatmap of tips data set")
plt.show()


# Iris Dataset

# In[33]:


iris = sns.load_dataset("iris")
print(iris)


# In[35]:


sns.pairplot(iris)
plt.title("Pairplot for iris dataset")
plt.show()


# In[ ]:



sns.heatmap(iris.corr(numeric_only=True) , fmt='.2f' , cmap='coolwarm' , annot=True , linewidths=0.5)
plt.title("Heatmap for iris dataset")
plt.show()

