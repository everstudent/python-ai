import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
model = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)
model.fit(x_train, y_train.values[:,0])

y_pred = model.predict(x_test)

print(model.feature_importances_)
print(np.sum(model.feature_importances_))
features = pd.DataFrame({'importance': model.feature_importances_, 'name': boston.feature_names})
print(features.sort_values('importance', ascending=False).head(2))

#     importance   name
# 12    0.415847  LSTAT
# 5     0.402682     RM
