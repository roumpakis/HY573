# -*- coding: utf-8 -*-
"""
For the purposes of MAM T5.02 - Numerical Methods for Optimization
University of Crete - Mathematics and Applied Mathematics Department
Winter 2023

@author: Lefteris Polychronakis
@ID:     math6090

@under professor: Panagiotis Chatzipantelidis

@brief: This file contains the algorithms of some methods for Optimization,
        namely: Nesterov Accelerated Gradient Descent with non fixed param,
                Nesterov Accelerated Gradient Descent with fixed param,
                Gradient Descent with Momentum, Vanilla Gradient Descent
        They're intended to be used for plotting.
"""

import numpy as np



def armijo_nesterov(func, dfunc, x, y, a_init = None):
    """
    Armijo line search algorithm adapted for Nesterov's method.

    Parameters:
    - func: The objective function to be minimized.
    - dfunc: The derivative of the objective function.
    - x: The current point in the parameter space.
    - y: The point to perform the line search from (usually the lookahead point in Nesterov's method).
    - a_init: Initial step size (optional).

    Returns:
    - The calculated step size.
    """
    a = 1
    if a_init is not None:
        a = a_init
 
    xnew = y - a * dfunc(y)
    dxnorm = np.linalg.norm(x - xnew) ** 2
    fx = func(x)
    dot = np.dot(dfunc(x),(xnew - x)) 
    l = fx + dot + 1/(2*a) * dxnorm
    
    cond = func( xnew ) > l

    ctr = 0
    while cond:
        a = a / 2
        xnew = y - a * dfunc(y)
        dxnorm = np.linalg.norm(x - xnew) ** 2
        fx = func(x)
        dot = np.dot(dfunc(x),(xnew - x)) 
        l = fx + dot + 1/(2*a) * dxnorm
        
        cond = func( xnew ) > l

        # loop control
        ctr += 1
        if cond == False:
            ctr = 0
        if ctr > 99 or a < 1e-32:
            return None

    return a

def armijo_nesterov2(func, dfunc, x, y, a_init = None):
    """
    A variant of the Armijo line search algorithm for Nesterov's method.

    Parameters:
    - Same as armijo_nesterov.

    Returns:
    - The calculated step size or None if line search fails.
    """
    a = 1
    if a_init is not None:
        a = a_init
 
    xnew = y
    dxnorm = np.linalg.norm(x - xnew) ** 2
    fx = func(x)
    dot = np.dot(dfunc(x),(xnew - x)) 
    l = fx + dot + 1/(2*a) * dxnorm
    
    cond = func( xnew ) > l

    ctr = 0
    while cond:
        a = a / 2
        xnew = y
        dxnorm = np.linalg.norm(x - xnew) ** 2
        fx = func(x)
        dot = np.dot(dfunc(x),(xnew - x)) 
        l = fx + dot + 1/(2*a) * dxnorm
        
        cond = func( xnew ) > l

        # loop control
        ctr += 1
        if cond == False:
            ctr = 0
        if ctr > 99 or a < 1e-32:
            print("yes")
            return None

    return a

def nesterov_momentum(func, dfunc, t, lamda=None, x_init=None,
                      line_search=True, dec_stepsize = True, max_iter=4000,
                      tol=1e-6, stop_criterion=0):
    """
    Nesterov Accelerated Gradient Descent (NAGD) with optional line search.

    Parameters:
    - func: The objective function to be minimized.
    - dfunc: The derivative of the objective function.
    - t: Initial step size.
    - lamda: Momentum term (optional).
    - x_init: Initial starting point (optional).
    - line_search: Whether to use line search.
    - dec_stepsize: Whether to decrease the step size over iterations.
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for stopping criterion.
    - stop_criterion: The criterion for stopping (0, 1, 2, or 3).

    Returns:
    - A history of points visited during the optimization process.
    """
    if x_init is None:
        x_init = np.random.rand(len(x_init))

    lamda_i = lamda_i1 = 0
    x_i = x_i1 = y_i = x_init
    x_history = [x_i]

    t_init  = t
    
    for i in range(max_iter):
        if line_search and dec_stepsize:
            t = armijo_nesterov(func, dfunc, x_i, y_i, a_init = t)
            if t is None:
                break
        elif line_search and not dec_stepsize:
            t = armijo_nesterov(func, dfunc, x_i, y_i, a_init = t_init)
            if t is None:
                break

        x_i1 = y_i - t * dfunc(y_i)
        if lamda is None:
            lamda_i1 = (1 + np.sqrt(1 + 4 * lamda_i**2)) / 2
            y_i1 = x_i1 + ((lamda_i - 1) / lamda_i1) * (x_i1 - x_i)
        else:
            y_i1 = x_i1 + (lamda - 1) / (lamda + 1) * (x_i1 - x_i)

        x_history.append(x_i1)

        # Stopping criterion
        if stop_criterion == 0:
            criterion = np.linalg.norm(func(x_i1) - func(x_i))
        elif stop_criterion == 1:
            criterion = np.linalg.norm(x_i1 - x_i)
        elif stop_criterion == 2:
            criterion = np.linalg.norm(dfunc(x_i1))
        elif stop_criterion == 3:
            criterion = np.linalg.norm(func(x_i1))

        if criterion < tol and i > 1:
            break

        x_i = x_i1
        y_i = y_i1
        lamda_i = lamda_i1

    # Print statements outside the loop to handle all cases
    grad_norm = np.linalg.norm(dfunc(x_i1))
    status = "converged" if criterion < tol else "did not converge"
    ls_status = "w/o line search" if not line_search else ""
    param_status = "w const. param." if lamda is not None else ""

    print(f"Nesterov {ls_status} {param_status} {status} after {i+1} iterations:")
    print(f"Min value: {func(x_i1)} Gradient Norm: {grad_norm}\n")

    return x_history



def gradient_descent(func, dfunc, lr=0.1, x_init=None, max_iter=50000,
                     tol=1e-6, line_search=False):
    """
    Vanilla Gradient Descent with optional line search.

    Parameters:
    - Same as nesterov_momentum except no lamda and dec_stepsize.

    Returns:
    - A history of points visited during the optimization process.
    """
    
    if x_init is None:
        x_init = np.random.rand(len(x_init))

    x_i = x_init
    x_history = [x_i]

    for i in range(max_iter):
        if line_search:
            lr = armijo_nesterov(func, dfunc, x_i, x_i, lr)
            if lr is None:
                break

        x_i1 = x_i - lr * dfunc(x_i)
        x_history.append(x_i1)

        grad_norm = np.linalg.norm(func(x_i1) - func(x_i))
        if grad_norm < tol:
            print(f"Vanilla Gradient Descent Converged after {i+1} iterations:")
            print(f"Min value: {func(x_i1)} Gradient Norm: {grad_norm}\n")
            return x_history

        x_i = x_i1

    return x_history


def gradient_descent_with_momentum(func, dfunc, beta=0.9, lr=0.1, x_init=None,
                                   max_iter=5000, tol=1e-6, line_search=False):
    """
    Gradient Descent with Momentum.

    Parameters:
    - func: The objective function to be minimized.
    - dfunc: The derivative of the objective function.
    - beta: Momentum coefficient.
    - lr: Learning rate or initial step size.
    - x_init: Initial starting point (optional).
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for stopping criterion.
    - line_search: Whether to use line search.

    Returns:
    - A history of points visited during the optimization process.
    """
    if x_init is None:
        x_init = np.random.rand(len(x_init))

    x_i = x_init
    v_i = np.zeros_like(x_i)
    x_history = [x_i]

    for i in range(max_iter):
        if line_search:
            lr_temp = armijo_nesterov(func, dfunc, x_i, x_i, lr)
            if lr_temp is None:
                break
            lr = lr_temp

        v_i = beta * v_i + lr * dfunc(x_i)
        x_i1 = x_i - v_i
        x_history.append(x_i1)

        grad_norm = np.linalg.norm(func(x_i1) - func(x_i))
        if grad_norm < tol:
            print(f"Gradient Descent with Momentum Converged after {i+1} iterations:")
            print(f"Min value: {func(x_i1)} Gradient Norm: {grad_norm}\n")
            return x_history

        x_i = x_i1

    return x_history
