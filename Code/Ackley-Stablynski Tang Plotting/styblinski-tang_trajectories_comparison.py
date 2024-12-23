# -*- coding: utf-8 -*-
"""
For the purposes of MAM T5.02 - Numerical Methods for Optimization
University of Crete - Mathematics and Applied Mathematics Department
Winter 2023

@author: Lefteris Polychronakis
@ID:     math6090

@under professor: Panagiotis Chatzipantelidis

@brief: This file is used for plotting trajectories of optimization algorithms
        on Styblinski-Tang Function.
"""

import optimization_methods as tools
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("default")

def styblinski_tang_2d(x):
    return 0.5 * (np.sum(x**4 - 16 * x**2 + 5 * x) + np.sum(x**4 - 16 * x**2 + 5 * x))


# Gradient of Styblinski-Tang function for n = 2
def styblinski_tang_gradient_2d(x):
    gradient = 2 * (x**3 - 8 * x + 5/4)
    return gradient


if __name__ == "__main__":
    # Initial guess
    initial_guess = np.array([-4,4])
    
    # Optimize the Styblinski-Tang function using different methods
    nesterov_optimization_history = tools.nesterov_momentum(
        styblinski_tang_2d,
        styblinski_tang_gradient_2d,
        x_init = initial_guess,
        t = 0.03, #0.065721 global
        dec_stepsize= False
    )
    
    const_param_optimization_history = tools.nesterov_momentum(
        styblinski_tang_2d,
        styblinski_tang_gradient_2d,
        x_init = initial_guess,
        t = 0.025,
        lamda = 37/3,# (l-1)/(l+1) = percentage, here percentage = 85%
                      # in order to compare with GD w momentum
        line_search = False,
    )
    
    wo_ls_optimization_history = tools.nesterov_momentum(
        styblinski_tang_2d,
        styblinski_tang_gradient_2d,
        x_init = initial_guess,
        t = 0.03,
        line_search = False
    )
    
    momentum_optimization_history = tools.gradient_descent_with_momentum(
        styblinski_tang_2d,
        styblinski_tang_gradient_2d,
        x_init = initial_guess,
        beta=0.85,
        lr=0.025,
        line_search = True
    )
    
    
    # Create meshgrid 
    cube = 5
    x = np.linspace(-cube, cube, 100)
    y = np.linspace(-cube, cube, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[styblinski_tang_2d(np.array([xi, yi])) for xi in x] for yi in y])
    
    # Plot the Styblinski-Tang function contours
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels = 25, cmap = 'magma')
    
    latex_title_styblinski_tang = r'$f(x, y) = 0.5 \left((x^4 - 16x^2 + 5x) + (y^4 - 16y^2 + 5y)\right)$'
    
    # plt.xlabel(r'$x$')
    # plt.ylabel(r'$y$')
    
    # Plot the optimization paths for Styblinski-Tang function
    nesterov_optimization_history = np.array(nesterov_optimization_history)
    const_param_optimization_history = np.array(const_param_optimization_history)
    wo_ls_optimization_history = np.array(wo_ls_optimization_history)
    momentum_optimization_history = np.array(momentum_optimization_history)
    
    plt.plot(nesterov_optimization_history[:, 0],
              nesterov_optimization_history[:, 1],
              color = 'red', marker = 'x', linestyle = 'dashed',
              markersize = 6, label='NAGD w LS & varying param Trajectory')
    
    plt.plot(const_param_optimization_history[:, 0],
              const_param_optimization_history[:, 1],
              color = 'green', marker = 'x', linestyle = 'dashed',
              markersize = 6, label='NAGD fixed param w\o LS Trajectory')
    
    plt.plot(wo_ls_optimization_history[:, 0],
              wo_ls_optimization_history[:, 1],
              color = 'blue', marker = 'x', linestyle = 'dashed',
              markersize = 6, label = 'NAGD w/o LS Trajectory')
    
    plt.plot(momentum_optimization_history[:, 0],
              momentum_optimization_history[:, 1],
              color = 'deepskyblue', marker = 'x', linestyle = 'dashed',
              markersize = 6, label = 'GD w Momentum Trajectory')
    

    plt.legend(loc = "upper right")
    plt.savefig("COMPARISON.png", dpi=300)

    plt.show()