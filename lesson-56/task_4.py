import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score


# data load
pd.options.display.max_columns = 100
df = pd.read_csv('/tmp/creditcard.csv', sep=',')
print(df.value_counts())
print(df.info())
print(df.head(10))


# prepare data
x = df.copy()
x.drop(['Class'], inplace=True, axis=1)
print(x.shape)

y = pd.Series(df['Class'])
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100, stratify=y);


# model grid
parameters = [{'n_estimators': [10, 15], 'max_features': np.arange(3, 5), 'max_depth': np.arange(4, 7)}]
clf = GridSearchCV(estimator=RandomForestClassifier(random_state=100), param_grid=parameters, scoring='roc_auc', cv=3)
clf.fit(x_train, y_train)


# results
print(clf.best_params_)
# {'max_depth': 6, 'max_features': 3, 'n_estimators': 15}

y_pred_proba = clf.predict_proba(x_test)[:,1]
auc = roc_auc_score(y_test, y_pred_proba)
print(auc)

# 0.9462664156037156
