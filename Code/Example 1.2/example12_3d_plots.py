# -*- coding: utf-8 -*-
"""
For the purposes of MAM T5.02 - Numerical Methods for Optimization
University of Crete - Mathematics and Applied Mathematics Department
Winter 2023

@author: Lefteris Polychronakis
@ID:     math6090

@under professor: Panagiotis Chatzipantelidis

@brief: This is a reproduction of Example 1.2 Noel J. Walkington's paper
        "Nesterov’s Method for Convex Optimization"
"""

import numpy as np
import matplotlib.pyplot as plt

def example12(x, y):
    return (np.log(1+x**2))**2 + 10* y**2

plt.style.use('default')

# Plot the function itself in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r'$f(x,y) = \log (1+x^2)^2 + 10y^2$')
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = example12(X, Y)
ax.plot_surface(X, Y, Z, cmap='magma', alpha = 0.8)

# nesterov_x = np.array(nesterov_values)[:, 0]
# nesterov_y =  np.array(nesterov_values)[:, 1]
# nesterov_z = example12(nesterov_x, nesterov_y)
# ax.plot(nesterov_x, nesterov_y, nesterov_z, color='orange', marker='x',
#         linestyle='-', label='Nesterov Trajectory', markersize=6)

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$f(x,y)$')

plt.savefig('function12', dpi = 300, bbox_inches='tight')
plt.show()