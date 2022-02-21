import numpy as np


# 1. task
a = np.array([
    [1, 6],
    [2, 8],
    [3, 11],
    [3, 10],
    [1, 7],
])

mean_a = a.mean(axis=0)
print(mean_a)


# 2. task
a_centered = a - mean_a
print(a_centered)


# 3. task
col0 = a_centered[:,0]
col1 = a_centered[:,1]

a_centered_sp = col0.dot(col1) / (len(col0) - 1)
print(a_centered_sp)


# 4. task
cov_matrix = np.cov(np.transpose(a))
print(cov_matrix[0,1])
