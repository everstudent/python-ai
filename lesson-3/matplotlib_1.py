import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline


# data
x = [1, 2, 3, 4, 5, 6, 7]
y = [3.5, 3.8, 4.2, 4.5, 5, 5.5, 7]


# plot settings
fg, ax = plt.subplots(nrows = 1, ncols = 2)
ax1, ax2 = ax.flatten()

# linear
ax1.plot(x, y)

# scatter
ax2.scatter(x, y)


plt.show()
