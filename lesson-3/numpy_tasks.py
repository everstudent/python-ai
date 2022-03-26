import numpy as np

# 1
a = np.linspace(12, 24, 12, endpoint=False)
print(a)


# 2
r1 = np.reshape(a, (3,4))
print(r1)

r2 = np.reshape(a, (2,6))
print(r2)

r3 = np.reshape(a, (4,3))
print(r3)

r4 = np.reshape(a, (6,2))
print(r4)

r5 = np.reshape(a, (12,1))
print(r5)


# 3
ar1 = np.reshape(a, (3,-1))
print(ar1)

ar2 = np.reshape(a, (2,-1))
print(ar2)

ar3 = np.reshape(a, (4,-1))
print(ar3)

ar4 = np.reshape(a, (6,-1))
print(ar4)

ar5 = np.reshape(a, (12,-1))
print(ar5)


# 4
# Можно, одномерный массив - это вектор


# 5
rng = np.random.default_rng()
a = rng.standard_normal((3,4))
print(a)
print(a.size)
b = a.flatten()
print(b)
print(b.size)


# 6
a = np.arange(20, 0, -2)
print(a)


# 7
b = np.arange(20, 1, -2)
print(b)

print(np.equal(a, b))


# 8
a = np.zeros((2, 2))
print(a)

b = np.ones((3, 2))
print(b)

c = np.vstack((a,b))
print( c )
print( c.size )


# 9
a = np.arange(0, 12)
A = np.reshape(a, (4, 3))
At = np.transpose(A)
print(At)

B = np.matmul(A, At)
print(B)
print(B.shape)

# Обратную матрицу вычислить невозможно, т.к. определитель матрицы равен нулю
Bdet = np.linalg.det(B)
print(Bdet)
# Bt = np.linalg.inv(B)
# print(Bt)


# 10
np.random.seed(42)


# 11
c = np.arange(0, 16)
print(c)


# 12
C = c.reshape((4,4))
print(C)

D = C * 10 + B
print(D)

Ddet = np.linalg.det(D)
print(Ddet)

Drank = np.linalg.matrix_rank(D)
print(Drank)

Dinv = np.linalg.inv(D)
print(Dinv)


# 13
D_ = np.where(Dinv <= 0, 0, 1)
E = np.where(D_ == 0, C, B)
print(E)
