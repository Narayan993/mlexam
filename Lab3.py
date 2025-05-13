#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[2]:


iris = load_iris()
x = iris.data
y = iris.target
target_names = iris.target_names

scaler= StandardScaler()
x_standardized = scaler.fit_transform(x)#check for x

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_standardized)
pca_df = pd.DataFrame(data=x_pca , columns=['Principal Component 1' ,"Principal Component 2"])
pca_df['Target'] = y
plt.figure(figsize=(8,6))
# plt.show()


# In[3]:


for target , color , label in zip(range(len(target_names)) , ['r','g','b'] , target_names):
    plt.scatter(
        pca_df.loc[pca_df['Target'] == target , 'Principal Component 1'] ,
        pca_df.loc[pca_df['Target']==target , 'Principal Component 2'],
        color = color,
        alpha=0.5,
        label =label
    )
plt.title('PCA of Iris Dataset (2 Components)' , fontsize = 16)
plt.ylabel('Principal Component 2',fontsize = 12)
plt.xlabel('Principal Component 1',fontsize = 12)
plt.legend(title='Target' , loc = 'best')
plt.grid(alpha=0.3)
plt.show()

