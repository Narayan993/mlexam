import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score
def linear_regression_boston():
    boston = fetch_openml(name="boston" , version=1 , as_frame=True)
    x = boston.data.to_numpy()
    y = boston.target.to_numpy()

    x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)
    ling_reg = LinearRegression()
    ling_reg.fit(x_train , y_train)

    y_pred = ling_reg.predict(x_test)

    mse = mean_squared_error(y_test , y_pred)
    r2 = r2_score(y_test , y_pred)

    plt.figure(figsize=(10 , 6))
    plt.scatter(y_test , y_pred , alpha=0.6)
    plt.plot([min(y_test) , max(y_test)] , [min(y_test) , max(y_test)] , color="red" , linewidth=2)
    plt.xlabel("True values (Boston Housing)")
    plt.ylabel("Predicted Values")
    plt.title("Linear Regression on Boston Housing Dataset")
    plt.grid(True)
    plt.show()

    print(f"Linear Regression Results: ")
    print(f"Mean Squared Error  : {mse:.2f}")
    print(f"R^2 Score : {r2:.2f}")
def Polynomial_regression_auto_mpg():
    auto_mpg = fetch_openml(name="autoMpg" , version=1 , as_frame=True)
    data=auto_mpg.data
    target = auto_mpg.target.astype(float)

    data = data.dropna(subset = ["horsepower"])
    target = target.loc[data.index]

    x_hp = data[["horsepower"]].astype(float)
    y_mpg = target

    x_train , x_test , y_train , y_test = train_test_split(x_hp , y_mpg , test_size=0.2 , random_state=42)

    poly_features = PolynomialFeatures(degree=3)
    x_train_poly = poly_features.fit_transform(x_train)
    x_test_poly = poly_features.transform(x_test)

    lr_poly = LinearRegression()
    lr_poly.fit(x_train_poly , y_train)

    y_pred_poly = lr_poly.predict(x_test_poly)

    mse_poly = mean_squared_error(y_test , y_pred_poly)
    r2_poly = r2_score(y_test , y_pred_poly)

    x_test_sorted , y_test_sorted = zip(*sorted(zip(x_test.values.flatten() , y_test)))
    y_pred_sorted = lr_poly.predict(poly_features.transform(np.array(x_test_sorted).reshape(-1 , 1)))
    plt.figure(figsize=(10 , 6))
    plt.scatter(x_test , y_test , color="blue" , label="True Values" , alpha=0.6)
    plt.plot(x_test_sorted , y_pred_sorted, color='red' , label="Polynomial fit degree(3)" , linewidth = 2)
    plt.xlabel("Horsepower")
    plt.ylabel("Miles per galloon (MPG)")
    plt.title("Polynomial regression on Auto MPG Dataset")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Polynoial Regression Results: ")
    print(f"Mean Squared Error  : {mse_poly:.2f}")
    print(f"R^2 Score : {r2_poly:.2f}")
def run_models():
    linear_regression_boston()
    Polynomial_regression_auto_mpg()
run_models()