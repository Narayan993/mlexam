#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
from sklearn.neighbors import KNeighborsClassifier 


# In[3]:


np.random.seed(42)
x = np.random.rand(100)
labels = np.where(x[:50] <= 0.5, 1, 2)
x_train = x[:50].reshape(-1, 1) 
y_train = labels 
x_test = x[50:].reshape(-1, 1) 
k_values = [1, 2, 3, 4, 5, 20, 30] 
print("k-NN Classification Results:") 
for k in k_values:
   knn = KNeighborsClassifier(n_neighbors=k) 
   knn.fit(x_train, y_train)
   y_pred = knn.predict(x_test)
   print(f"\nk = {k}:") 
   print(f"Predicted Classes: {y_pred}") 

