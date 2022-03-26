import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# dataset
boston = load_boston()

x = pd.DataFrame(boston.data, columns=boston.feature_names)
print(x.info())

y = pd.DataFrame(boston.target, columns=['price'])
print(y.info())


# samples
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .30, random_state = 42)


# model
lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
print('R2:', r2_score(y_pred, y_test))
