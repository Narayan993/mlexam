import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
def load_and_preprocess_data():
    data = load_breast_cancer()

    x = data.data
    y = data.target

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled , y
def apply_kmeans(x , n_clusters = 2):
    kmeans = KMeans(n_clusters=n_clusters , random_state=42)
    kmeans.fit(x)
    return kmeans.cluster_centers_ , kmeans.labels_
def visualize_clusters(x , labels , centers):
    plt.figure(figsize=(8,6))
    plt.scatter(x[: , 0] , x[: , 1] , c=labels , cmap='viridis' , s=50)

    plt.scatter(centers[: , 0] , centers[: , 1] , c = 'red' , marker='x' , s=200 , label='Centroids')

    plt.title('K-Means clustering on wisconsin breast cancer dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
def main():
    x , y = load_and_preprocess_data()
    centers , labels = apply_kmeans(x , n_clusters=3)
    visualize_clusters(x , labels , centers)
if __name__ == "__main__":
    main()