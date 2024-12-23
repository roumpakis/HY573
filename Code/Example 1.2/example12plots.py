# -*- coding: utf-8 -*-
"""
University of Crete - Mathematics and Applied Mathematics Department
Winter 2023

@author: Lefteris Polychronakis
@ID:     math6090

@under professor: Panagiotis Chatzipantelidis

@brief: This is a reproduction of Example's 1.2 function 
        from Noel J. Walkington's paper "Nesterov’s Method for Convex
        Optimization"
"""
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')

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
    l = fx + a/2 * dfnorm
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
    
    armijo = False    
    if t is None:
        armijo = True
        t = .5
        
    lamda_i = 0
    x_i = y_i = x_init
    values = [x_i]
    
    for i in range(max_iter):
        if armijo:
            t = armijo_nesterov(func, dfunc, x_i, y_i, a_init = t)
                
        x_i1 = y_i - t * dfunc(*y_i)
        lamda_i1 = (1 + np.sqrt(1 + 4 * lamda_i**2)) / 2
        y_i1 = x_i1 + ((lamda_i - 1) / lamda_i1) * (x_i1 - x_i)
        
        values.append(x_i1)
        
        if abs(func(*values[-1]) - func(*values[-2])) < tol and i > 2:
            return values

        x_i = x_i1
        y_i = y_i1
        lamda_i = lamda_i1
    
    return values


def steepest_descent(func, dfunc, t = None, x_init = None, max_iter=1000,
                     tol = 1e-8):
    
    if x_init is None:
        x_init = np.random.rand(2)
        
    armijo = False    
    if t is None:
        armijo = True
        
    x_i = x_init
    values = [x_i]
    
    for i in range(max_iter):
        gradient = dfunc(*x_i)
        
        if armijo:
            t = armijo_line_search(func, dfunc, x_i, -gradient, 0.1)
            
        x_i1 = x_i - t * gradient
        
        values.append(x_i1)
        
        if abs(func(*values[-1]) - func(*values[-2])) < tol and i > 2:
            return values
        
        x_i = x_i1

    return values

def plot_iteration_values(func, values_list, labels=None):
    plt.figure(figsize=(8, 6))
    legend = ['Steepest Descent', 'Nesterov']
    
    for i, values in enumerate(values_list):
        iterations = list(range(len(values)))
        function_values = [func(*val) for val in values]
        
        linestyle = '-' if i == 0 else '--'  # Use dashed line for Nesterov's trajectory
        label = legend[i]
        plt.plot(iterations, function_values, linestyle=linestyle, label=label)

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value (log scale)')
    plt.title(r'$f(x,y) = \log (1+x^2)^2 + 10y^2$')
    plt.legend()
    plt.savefig('cost_per_iter.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_contour_with_trajectory(ax, func, values, method_color, labels=None):
    # Create a meshgrid for the contour plot
    x = np.linspace(-3.5, 3.5, 100)
    y = np.linspace(-3.5, 3.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    # Plot the contour plot
    contours = ax.contour(X, Y, Z, levels=15, cmap='magma')
    ax.clabel(contours, inline=True, fontsize=8)

    for i, traj in enumerate(values):
        x_values = [point[0] for point in traj]
        y_values = [point[1] for point in traj]

        label = f'Trajectory {i+1}' if labels is None else labels[i]
        ax.plot(x_values, y_values, marker='x', linestyle='-', label=label,
                color=method_color, markersize=6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()


nesterov_values = nesterov_momentum(example12, example12_derivative,
                            x_init = np.array([1,1]))

steepest_descent_values = steepest_descent(example12, example12_derivative,
                            x_init = np.array([1,1]))

plot_iteration_values(example12, [steepest_descent_values, nesterov_values])


# Plot trajectories on separate subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

plot_contour_with_trajectory(axes[0], example12, [nesterov_values], 
                             method_color='orange', labels=['Nesterov'])

plot_contour_with_trajectory(axes[1], example12, [steepest_descent_values],
                             method_color='blue', labels=['Steepest Descent'])
plt.tight_layout()
plt.savefig('contours', dpi = 300, bbox_inches='tight')
plt.show()