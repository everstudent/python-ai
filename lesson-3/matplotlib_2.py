import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline


# data
t = np.linspace(0, 10, 51)
f = np.cos(t)


# chart
plt.plot(t, f, color = 'green')
plt.title('График f(t)')
plt.xlabel('Значения t')
plt.ylabel('Значения f')
plt.axis([0.5, 9.5, -2.5, 2.5])
plt.show()
