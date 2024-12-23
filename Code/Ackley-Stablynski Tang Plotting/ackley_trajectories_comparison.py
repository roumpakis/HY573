# -*- coding: utf-8 -*-
"""
For the purposes of MAM T5.02 - Numerical Methods for Optimization
University of Crete - Mathematics and Applied Mathematics Department
Winter 2023

@author: Lefteris Polychronakis
@ID:     math6090

@under professor: Panagiotis Chatzipantelidis

@brief: This file is used for plotting trajectories of optimization algorithms
        on Ackleys function.
"""

import optimization_methods as tools
import numpy as np
import matplotlib.pyplot as plt

def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.exp(1)

def ackley_gradient(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)

    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))

    grad = -a * (-b / np.sqrt(n)) * np.exp(-b * np.sqrt(sum1 / n)) * (x / np.sqrt(sum1))
    grad -= (1 / n) * np.sin(c * x) * (2 * np.pi / n) * np.exp(sum2 / n)

    return grad

if __name__ == "__main__":
    
    initial_guess = np.array([3.9, 2])
    
    
    nesterov_optimization_history = tools.nesterov_momentum(
        ackley,
        ackley_gradient,
        x_init = initial_guess,
        t = 0.2,
        dec_stepsize = False
    )
    
    const_param_optimization_history = tools.nesterov_momentum(
        ackley,
        ackley_gradient,
        x_init = initial_guess,
        t = 0.13,
        lamda = 5.6,
        line_search = False
    )
    
    wo_ls_optimization_history = tools.nesterov_momentum(
        ackley,
        ackley_gradient,
        x_init = initial_guess,
        t = 0.1,
        line_search = False
    )
    
    momentum_optimization_history = tools.gradient_descent_with_momentum(
        ackley,
        ackley_gradient,
        x_init = initial_guess,
        beta = 0.7,
        lr = 0.1
    )
    
    vanilla_optimization_history = tools.gradient_descent(
        ackley,
        ackley_gradient,
        x_init = initial_guess,  
        lr = 0.1  
    )
    
    
    # Plot the contour of the Ackley function
    cube = 4.2
    x = np.linspace(-cube, cube, 100)
    y = np.linspace(-cube, cube, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[ackley(np.array([xi, yi])) for xi in x] for yi in y])
    
    plt.figure(figsize = (8, 6))
    plt.contour(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 8), cmap='magma')
    
    latex_part1 = r'$f(x_1, x_2) = -a \cdot \exp\left(-b \cdot \sqrt{\frac{1}{2} \sum_{i=1}^{2} x_i^2}\right)$'
    latex_part2 = r'$ - \exp\left(\frac{1}{2} \sum_{i=1}^{2} \cos(c \cdot x_i)\right) + a + \exp(1)$'
    
    # Concatenate the parts to form the complete LaTeX title
    latex_title = latex_part1 + latex_part2
    #plt.title(latex_title)
    # plt.xlabel(r'$x_1$')
    # plt.ylabel(r'$x_2$')
    
    # Plot the optimization paths
    nesterov_optimization_history = np.array(nesterov_optimization_history)
    const_param_optimization_history = np.array(const_param_optimization_history)
    wo_ls_optimization_history = np.array(wo_ls_optimization_history)
    momentum_optimization_history = np.array(momentum_optimization_history)
    vanilla_optimization_history = np.array(vanilla_optimization_history)
    wo_ls_optimization_history = np.array(wo_ls_optimization_history)
    
    plt.plot(nesterov_optimization_history[:, 0],
              nesterov_optimization_history[:, 1],
              color = 'red', marker = 'x', linestyle = 'dashed',
              markersize = 6, label='NAGD w LS & varying param Trajectory')
    
    plt.plot(const_param_optimization_history[:, 0],
              const_param_optimization_history[:, 1],
              color = 'green', marker = 'x', linestyle = 'dashed',
              markersize = 6,  label='NAGD fixed param w\o LS Trajectory')
    
    plt.plot(wo_ls_optimization_history[:, 0],
              wo_ls_optimization_history[:, 1],
              color = 'blue', marker = 'x', linestyle = 'dashed',
              markersize = 6, label = 'NAGD w/o LS Trajectory')
    
    plt.plot(momentum_optimization_history[:, 0],
              momentum_optimization_history[:, 1],
              color = 'deepskyblue', marker = 'x', linestyle = 'dashed',
              markersize = 6, label = 'GD w Momentum Trajectory')
    
    plt.plot(vanilla_optimization_history[:, 0],
              vanilla_optimization_history[:, 1],
              color = 'orange', marker = 'x', linestyle = 'dashed',
              markersize = 6, label = 'Vanilla GD Trajectory')
    
    
    
    plt.legend(loc = 'upper left') 
    plt.show()