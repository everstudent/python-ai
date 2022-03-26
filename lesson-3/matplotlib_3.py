import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline


# data
x = np.linspace(-3, 3, 51)
y1 = x**2
y2 = 2*x + 0.5
y3 = -3*x - 1.5
y4 = np.sin(x)


# plot
fig, ax = plt.subplots(nrows = 2, ncols = 2)
ax1, ax2, ax3, ax4 = ax.flatten()

fig.set_size_inches(8, 6)
fig.subplots_adjust(wspace=.3, hspace=.3)

ax1.plot(x, y1)
ax1.set_xlim([-5, 5])
ax1.set_title('График y1')

ax2.plot(x, y2)
ax2.set_title('График y2')

ax3.plot(x, y3)
ax3.set_title('График y3')

ax4.plot(x, y4)
ax4.set_title('График y4')

plt.show()
