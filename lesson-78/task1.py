import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.manifold import TSNE

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


# init TSNE model
tsne = TSNE(n_components=2, learning_rate=250, random_state=42)

# train model
X_train_tsne = tsne.fit_transform(X_train_scaled)

# check features amount
print(X_train_scaled.shape, '->', X_train_tsne.shape)

# plot features
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1])
plt.savefig('task1.png')
plt.show()