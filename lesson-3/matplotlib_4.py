import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %matplotlib inline


# load data
df = pd.read_csv('creditcard.csv', sep=',')


# plot classes
vc = pd.Series(df['Class'].value_counts())
vc.plot(kind='bar') # standard scale
plt.show()

vc.plot(kind='bar', logy=True) # log scale
plt.show()


# plot V1
plt.hist(df[df['Class']==1]['V1'], density=df[df['Class']==1]['V1'].max(), alpha=0.5, bins=20, color='gray')
plt.hist(df[df['Class']==0]['V1'], density=df[df['Class']==0]['V1'].max(), alpha=0.5, bins=20, color='red')

plt.legend(labels=["Class 1", "Class 2"])
plt.xlabel('V1');

plt.show();
