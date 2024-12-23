# -*- coding: utf-8 -*-
"""
For the purposes of MAM T5.02 - Numerical Methods for Optimization
University of Crete - Mathematics and Applied Mathematics Department
Winter 2023

@author: Lefteris Polychronakis
@ID:     math6090

@under professor: Panagiotis Chatzipantelidis

@brief: This is a reproduction of Example 1.2 from Noel J. Walkington's paper
        "Nesterov’s Method for Convex Optimization"
"""

import numpy as np


def example12_derivative(x, y):
    df_dx = 4 * x * np.log(1 + x**2) / (1 + x**2)
    df_dy = 20 * y
    return np.array([df_dx, df_dy])

def example12(x, y):
    return (np.log(1+x**2))**2 + 10* y**2


def armijo_line_search(func, dfunc, x, p, c, a_init = None):
    
    a = 1
    if a_init is not None:
        a = a_init
    
    pf = np.dot(p, dfunc(*x))
    fx = func(*x)
    l = fx + a * c * pf
    cond = func(*(x + a * p)) > l

    while cond:
        a = a / 2
        l = fx + a * c * pf
        cond = func(*(x + a * p)) > l

    return a

def armijo_nesterov(func, dfunc, x, y, a_init = None):
    
    a = 1
    if a_init is not None:
        a = a_init
        
    dxnorm = np.linalg.norm(y - a * dfunc(*y) - x) ** 2
    fx = func(*x)
    dot = np.dot(dfunc(*x),(y - a * dfunc(*y)) - x) 
    l = fx + dot + 1/(2*a) * dxnorm
    cond = func(*(y - a * dfunc(*y))) > l
    ctr = 0
    
    while cond:
        a = a / 2
            
        dxnorm = np.linalg.norm(y - a * dfunc(*y) - x) ** 2
        fx = func(*x)
        dot = np.dot(dfunc(*x),(y - a * dfunc(*y)) - x) 
        l = fx + dot + 1/(2*a) * dxnorm
        
        cond = func(*(y - a * dfunc(*y))) > l

        # loop control
        ctr += 1
        if cond == False:
            # print()
            ctr = 0
        if ctr > 99 or a < 1e-32:
            return None

    return a

def armijo_nesterov2(func, dfunc, x, y, a_init = None):
    
    a = 1
    if a_init is not None:
        a = a_init
        
    dfnorm = np.linalg.norm(dfunc(*x)) ** 2
    fx = func(*x)
    l = fx - a/2 * dfnorm
    cond = func(*(y - a * dfunc(*y))) > l
    ctr = 0
    while cond:
        a = a / 2
        l = fx - a/2 * dfnorm
        cond = func(*(y - a * dfunc(*y))) > l

        # loop control
        ctr += 1
        if cond == False:
            # print()
            ctr = 0
        if ctr > 99 or a < 1e-32:
            return None

    return a

def nesterov_momentum(func, dfunc, t = None, x_init = None, max_iter = 1000, 
                      tol = 1e-8):
    
    if x_init is None:
        x_init = np.random.rand(2)
    
    if t is None:
        armijo = True
        t = .5
        
    lamda_i = 0
    x_i = y_i = x_init
    
    for i in range(max_iter):
        if armijo:
            t = armijo_nesterov(func, dfunc, x_i, y_i, a_init = t)
            # print(t)
            if t == None:
                print("gamw")
                return i
            
        x_i1 = y_i - t * dfunc(*y_i)
        lamda_i1 = (1 + np.sqrt(1 + 4 * lamda_i**2)) / 2
        y_i1 = x_i1 + ((lamda_i - 1) / lamda_i1) * (x_i1 - x_i)

        if abs(func(*x_i1) - func(*x_i)) < tol and i > 2:
            print(f"Nesterov Converged after {i+1} iterations:")
            print("Min value: {0} Gradient Norm: {1}\n"
                  .format(func(*x_i1), np.linalg.norm(dfunc(*x_i1))))
            return i

        x_i = x_i1
        y_i = y_i1
        lamda_i = lamda_i1
    
    return i


def steepest_descent(func, dfunc, t = None, x_init = None, max_iter=1000,
                     tol = 1e-8):
    
    if x_init is None:
        x_init = np.random.rand(2)
        
    armijo = False    
    if t is None:
        armijo = True
        
    x_i = x_init
    
    for i in range(max_iter):
        gradient = dfunc(*x_i)
        
        if armijo:
            t = armijo_line_search(func, dfunc, x_i, -gradient, 0.5)
            # print(t)
            
        x_i1 = x_i - t * gradient
        # Convergence check
        if abs(func(*x_i1) - func(*x_i)) < tol:
            print(f"SD Converged after {i+1} iterations.")
            print("Min value: {0:} Gradient Norm: {1:}\n"
                  .format(func(*x_i1), np.linalg.norm(dfunc(*x_i1))))
            return i
        
        x_i = x_i1

    return i

    
resultN = nesterov_momentum(example12, example12_derivative, x_init = [1,1])
resultSD = steepest_descent(example12, example12_derivative, x_init = [1,1])