import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# suppress warnings
import warnings
warnings.filterwarnings("ignore")


# load data
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target


# split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# scale
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# TSNE downsizing
tsne = TSNE(n_components=2, learning_rate=250, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_scaled)
X_test_tsne = tsne.fit_transform(X_test_scaled)


# KMeans clustering
km = KMeans(n_clusters=3, random_state=42, max_iter=100)
km_trained = km.fit_predict(X_train_scaled)
km_test = km.predict(X_test_scaled)


# Plotting
plt.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c=km_test)


# Average price & CRIM
for i in range(0, 3):
  crim = X_test[km_test == i]['CRIM'].mean()
  price = y_test[km_test == i].mean()
  plt.text(X_test_tsne[:, 0][km_test == i].mean(), X_test_tsne[:, 1][km_test == i].mean(), f"CRIM: {crim}", c='r',fontsize=8)
  plt.text(X_test_tsne[:, 0][km_test == i].mean(), X_test_tsne[:, 1][km_test == i].mean()+0.5, f"Price: {price}", c='r',fontsize=8)

plt.savefig('task3.png')
plt.show()