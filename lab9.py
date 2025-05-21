import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
def load_and_preprocess_data():
    faces_data = datasets.fetch_olivetti_faces(shuffle=True , random_state=42)
    x = faces_data.data 
    y = faces_data.target
    return x, y
def split_data(x , y , test_size = 0.2):
    x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=test_size , random_state=42)
    return x_train , x_test , y_train , y_test
def train_naive_bayes(x_train , y_train):
    nb_classifier = GaussianNB()
    nb_classifier.fit(x_train , y_train)
    return nb_classifier
def predict_and_evaluate(nb_classifier , x_test , y_test):
    y_pred = nb_classifier.predict(x_test)
    accuracy = accuracy_score(y_test , y_pred)
    return y_pred , accuracy
def visualize_predictions(x_test , y_pred , n_samples=5):
    plt.figure(figsize=(10,5))
    for i in range(n_samples):
        plt.subplot(1 , n_samples , i+1)
        plt.imshow(x_test[i].reshape(64,64) , cmap='gray')
        plt.title(f"pred : {y_pred[i]}")
        plt.axis("off")
    plt.show()
def main():
    x , y = load_and_preprocess_data()
    x_train , x_test , y_train , y_test = split_data(x , y)
    nb_classifier = train_naive_bayes(x_train , y_train)
    y_pred , accuracy = predict_and_evaluate(nb_classifier , x_test , y_test)
    print(f"Accuracy of the naive bayes classifier on the test set : {accuracy*100:.2f}%" )

    visualize_predictions(x_test , y_pred)
if __name__=="__main__":
    main()