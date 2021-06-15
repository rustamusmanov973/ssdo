
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X3, Y3 = np.meshgrid(x, y)
Z3 = f(X3, Y3)
            Z3.shape


            # Z2
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap=cm.coolwarm)
ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# surf = ax.plot_surface(X3, Y3, Z3, cmap=cm.coolwarm,
#                     linewidth=0, antialiased=False)
            # X3.shape, Y3.shape, Z3.shape, X2.shape, Y2.shape, Z2.shape

            # X3
            # X2
            # ax.set
            # Z2 = f(X2, Y2)
surf = ax.plot_surface(X2, Y2, Z2, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# Plot the surface.

# Customize the z axis.
# z_formatter = plt.ticker.ScalarFormatter(useOffset=True)
# ax.zaxis.set_major_formatter(z_formatter)

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


If you have not have expression for Z
https://stackoverflow.com/questions/25764088/python3plot-fx-y-preferably-using-matplotlib

#!/usr/bin/python3

import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy
from numpy.random import randn, shuffle
from scipy import linspace, meshgrid, arange, empty, concatenate, newaxis, shape


# =========================
## generating ordered data:

N = 128
x = sorted(randn(N))
y = sorted(randn(N))

X, Y = meshgrid(x, y)
Z = X**2 + Y**2


# =======================
## re-shaping data in 1D:

# flat and prepare for concat:
X_flat = X.flatten()[:, newaxis]
Y_flat = Y.flatten()[:, newaxis]
Z_flat = Z.flatten()[:, newaxis]

DATA = concatenate((X_flat, Y_flat, Z_flat), axis=1)

shuffle(DATA)

Xs = DATA[:,0]
Ys = DATA[:,1]
Zs = DATA[:,2]


# ====================================================
## plotting surface using X, Y and Z given as 1D data:

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)
fig.colorbar(surf)

title = ax.set_title("plot_trisurf: takes X, Y and Z as 1D")
title.set_y(1.01)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

fig.tight_layout()
fig.savefig('3D-reconstructing-{}.png'.format(N))
