#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing


# In[3]:


housing = fetch_california_housing()
df = pd.DataFrame(housing.data , columns=housing.feature_names)
print(df)


# In[5]:


plt.figure(figsize=(12,8))
df.hist(bins=30 , figsize=(12,8) , color='blue' , alpha = 0.7)
plt.suptitle("Histogram of numerical features " , fontsize=14)
plt.show()


# In[9]:


plt.figure(figsize=(12,8))
sns.boxplot(data=df)
plt.xticks(rotation = 45)
plt.title("Box Plots of numerical features " , fontsize=14)
plt.show()


# In[23]:


outlier_info = {}
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 - 1.5*IQR
    outliers = df[(df[col] < lower_bound ) | (df[col] > upper_bound)][col]
    outlier_info[col] = outliers.count()

print("\nOutlier Count per feature :")
for i , j in outlier_info.items():
    print(f'{i}:{j} outliers')


# In[ ]:




