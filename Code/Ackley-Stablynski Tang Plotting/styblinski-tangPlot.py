# -*- coding: utf-8 -*-
"""
For the purposes of MAM T5.02 - Numerical Methods for Optimization
University of Crete - Mathematics and Applied Mathematics Department
Winter 2023

@author: Lefteris Polychronakis
@ID:     math6090

@under professor: Panagiotis Chatzipantelidis

@brief: This file is used for plotting Styblinski-Tang Function in 3 dimensions.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')

def styblinski_tang_2d(x, y):
    return 0.5 * ((x**4 - 16*x**2 + 5*x) + (y**4 - 16*y**2 + 5*y))

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

Z = styblinski_tang_2d(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='magma')
plt.savefig("St_Tang_plot.png", dpi=300)

# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$')

plt.show()
