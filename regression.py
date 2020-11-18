import requests
import pandas
import scipy
import numpy as np
import sys

# from sklearn.linear_model import LinearRegression


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


class LR:

	def predict(self, X):
		return np.dot(X, self.W)

	def grad_descent(self, X, Y, lr):
		pred = self.predict(X)

		error = np.sqrt(np.mean((pred - Y) ** 2))

		grad = np.sqrt(np.mean((pred - Y)))
		self.W -= lr * grad

	def fit(self, X, Y, iter = 100000, lr = 0.01):
		self.W = np.zeros(X.shape[0])

		for i in range(iter):
			self.grad_descent(X, Y, lr);


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    ...
    # print(response.content)
    x = str(response.content)
    # print(x)
    x = x.split('\\n')
    # print(len(x))
    # print(x[2])
    a = x[0].split(',')
    p = x[1].split(',')
    a = a[1:]
    p = p[1:]
    a = np.array([float(i) for i in a])
    p = np.array([float(i) for i in p])
    print(a.shape)
    clf = LR()
    clf.fit(a, p)

    ans = []
    for i in area:
    	ans.append(i)
    return ans
    # print(a)

    return 0


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = np.array(list(validation_data.keys()))
    prices = np.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = np.sqrt(np.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
