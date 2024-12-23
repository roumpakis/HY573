# -*- coding: utf-8 -*-
"""
For the purposes of MAM T5.02 - Numerical Methods for Optimization
University of Crete - Mathematics and Applied Mathematics Department
Winter 2023

@author: Lefteris Polychronakis
@ID:     math6090

@under professor: Panagiotis Chatzipantelidis

@brief: This file is used for comparing various optimization algorithms on
        rotated hyper ellipsoid function which is a convex function
"""

import numpy as np
import matplotlib.pyplot as plt
import optimization_methods as tools

def rotated_hyper_ellipsoid(x):
    n = len(x)
    return sum(sum(x[:i+1])**2 for i in range(n))

def rotated_hyper_ellipsoid_gradient(x):
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        for j in range(i, n):
            grad[i] += sum(x[:j+1])
        grad[i] *= 2
    return grad


initial_guess = 1 * np.ones(10,)

nesterov_optimization_history = tools.nesterov_momentum(
    rotated_hyper_ellipsoid,
    rotated_hyper_ellipsoid_gradient,
    x_init = initial_guess,
    t = 1,
    stop_criterion= 0
)

const_param_optimization_history = tools.nesterov_momentum(
    rotated_hyper_ellipsoid,
    rotated_hyper_ellipsoid_gradient,
    x_init = initial_guess,
    t = 0.03,
    lamda = 30,
    # line_search = False,
)


momentum_optimization_history = tools.gradient_descent_with_momentum(
    rotated_hyper_ellipsoid,
    rotated_hyper_ellipsoid_gradient,
    x_init = initial_guess,
    beta=0.85,
    lr=0.02,
    line_search = True
)

vanilla_optimization_history = tools.gradient_descent(
    rotated_hyper_ellipsoid,
    rotated_hyper_ellipsoid_gradient,
    x_init = initial_guess,  
    lr = 0.02  ,
    line_search = True
)

# ###############################################################################
# ###############################################################################
# ###############################################################################
# ###############################################################################

# print(75*"*")
# print(75*"*")

initial_guess = 1 * np.ones(100,)

nesterov_optimization_history = tools.nesterov_momentum(
    rotated_hyper_ellipsoid,
    rotated_hyper_ellipsoid_gradient,
    x_init = initial_guess,
    t = 1
)

const_param_optimization_history = tools.nesterov_momentum(
    rotated_hyper_ellipsoid,
    rotated_hyper_ellipsoid_gradient,
    x_init = initial_guess,
    t = 0.03,
    lamda = 30,
    # line_search = False,
)


momentum_optimization_history = tools.gradient_descent_with_momentum(
    rotated_hyper_ellipsoid,
    rotated_hyper_ellipsoid_gradient,
    x_init = initial_guess,
    beta=0.85,
    lr=.00009, #0.05, 0.09, 1
    line_search = True
)

vanilla_optimization_history = tools.gradient_descent(
    rotated_hyper_ellipsoid,
    rotated_hyper_ellipsoid_gradient,
    x_init = initial_guess,  
    lr = 0.00005,
    line_search = True
)

