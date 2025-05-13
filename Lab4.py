#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
data ={
    "weather" : ["sunny" , "sunny" , "Rainy" , "sunny"],
    "Temperature" : ["Warm" , "Warm" , "cold" , "Warm"],
    "Humidity" : ["Normal" , "High" , "High" , "High"],
    "Wind" : ["Strong" , "Strong" , "Strong" , "Weak"],
    "PlayTennis" : ["Yes" , "Yes" , "No" , "Yes"]
}


# In[3]:


df = pd.DataFrame(data)
file_path =  "training_data.csv"
df.to_csv(file_path , index=False)
print(f"CSV file 'training_data.csv' has been created successfully!")
def find_s_algorithim(data):
    features = data.iloc[: , :-1].values
    labels = data.iloc[: , -1].values
    hypotheses = None
    for i,label in enumerate(labels):
        if label == "Yes":
            hypotheses = features[i].copy()
            break
    if hypotheses is None:
        return "No Positive examples found"
    for i , label in enumerate(labels):
        if label == "Yes":
            for j in range(len(hypotheses)):
                if hypotheses[j] != features[i][j]:
                    hypotheses[j] = "?"
    return hypotheses

file_path = "training_data.csv"
data = pd.read_csv(file_path)
print("Training data : ")
print(data)

final_hypothesis = find_s_algorithim(data)

print("\n Final hypotheses : ")
print(final_hypothesis)

