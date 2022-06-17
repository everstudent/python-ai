import numpy as np
import pandas as pd
import pickle
import os

from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d.axes3d import Axes3D

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as r2, mean_absolute_error as mae, mean_squared_error as mse
from sklearn.ensemble import GradientBoostingRegressor

import warnings
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_preds(train_true_values, train_pred_values, test_true_values, test_pred_values):
  print("Train R2:\t" + str(round(r2(train_true_values, train_pred_values), 3)))
  print("Test R2:\t" + str(round(r2(test_true_values, test_pred_values), 3)))

  plt.figure(figsize=(18,10))

  plt.subplot(121)
  sns.scatterplot(x=train_pred_values, y=train_true_values)
  plt.xlabel('Predicted values')
  plt.ylabel('True values')
  plt.title('Train sample prediction')

  plt.subplot(122)
  sns.scatterplot(x=test_pred_values, y=test_true_values)
  plt.xlabel('Predicted values')
  plt.ylabel('True values')
  plt.title('Test sample prediction')

  plt.show()


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# 1. Remove "price"
train_df.drop(['Price'], axis=1, inplace=True)

# 2. Remove "Id" as meaningless column
train_df.drop(['Id'], axis=1, inplace=True)

# 3. Research disticts
# train_df.plot(y='DistrictId')
# plt.show();

# Research Square & LifeSquare
# train_df.plot(y='Square')
# distribution is ok


train_df[train_df['LifeSquare'] >= 500].plot(y='LifeSquare')
print(train_df[train_df['LifeSquare'] >= 1000])
# distribution is ok


# train_df[train_df['LifeSquare'] != None ].plot(x='Square', y='LifeSquare')
plt.show();

print(train_df.info())
print(train_df)
print(train_df.describe())


#X = train_df
#y = test_df

#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=21)

#rf_model = RandomForestRegressor(random_state=21, criterion='mse')
#rf_model.fit(X_train, y_train)

#y_train_preds = rf_model.predict(X_train)
#y_test_preds = rf_model.predict(X_valid)

#evaluate_preds(y_train, y_train_preds, y_valid, y_test_preds)