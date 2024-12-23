# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:33:21 2024

@author: csdro
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import optimization_methods as tools  # Assuming this contains the optimization methods


def forward_kinematics(theta1, theta2, theta3, L1, L2, L3):
    """
    Compute the positions of the joints and end-effector given angles and lengths.
    """
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    
    x3 = x2 + L3 * np.cos(theta1 + theta2 + theta3)
    y3 = y2 + L3 * np.sin(theta1 + theta2 + theta3)
    
    return x1, y1, x2, y2, x3, y3


def objective_function_polar(thetas, L1, L2, L3, x_t, y_t):
    """
    Calculate the squared distance between the end-effector and the target point.
    """
    theta1, theta2, theta3 = thetas
    _, _, _, _, x3, y3 = forward_kinematics(theta1, theta2, theta3, L1, L2, L3)
    return (x3 - x_t)**2 + (y3 - y_t)**2


def gradients_polar(thetas, L1, L2, L3, x_t, y_t):
    """
    Compute the gradients of the squared distance with respect to theta1, theta2, and theta3.
    """
    theta1, theta2, theta3 = thetas
    x1, y1, x2, y2, x3, y3 = forward_kinematics(theta1, theta2, theta3, L1, L2, L3)

    dx3_dtheta1 = -L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2) - L3 * np.sin(theta1 + theta2 + theta3)
    dy3_dtheta1 = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2) + L3 * np.cos(theta1 + theta2 + theta3)
    dx3_dtheta2 = -L2 * np.sin(theta1 + theta2) - L3 * np.sin(theta1 + theta2 + theta3)
    dy3_dtheta2 = L2 * np.cos(theta1 + theta2) + L3 * np.cos(theta1 + theta2 + theta3)
    dx3_dtheta3 = -L3 * np.sin(theta1 + theta2 + theta3)
    dy3_dtheta3 = L3 * np.cos(theta1 + theta2 + theta3)

    grad_theta1 = 2 * (x3 - x_t) * dx3_dtheta1 + 2 * (y3 - y_t) * dy3_dtheta1
    grad_theta2 = 2 * (x3 - x_t) * dx3_dtheta2 + 2 * (y3 - y_t) * dy3_dtheta2
    grad_theta3 = 2 * (x3 - x_t) * dx3_dtheta3 + 2 * (y3 - y_t) * dy3_dtheta3

    return np.array([grad_theta1, grad_theta2, grad_theta3])


def wrap_and_optimize(L1, L2, L3, x_t, y_t, t, x_init, method):
    """
    Optimize the robot arm angles using different methods: GD, GDM, and NAGD.
    """
    if method == 'gd':
        history = tools.gradient_descent(objective_function_polar, gradients_polar, t, x_init=x_init)
    elif method == 'gdm':
        history = tools.gradient_descent_with_momentum(objective_function_polar, gradients_polar, t, x_init=x_init)
    elif method == 'nagd':
        history = tools.nesterov_momentum(objective_function_polar, gradients_polar, t, x_init=x_init)
    else:
        raise ValueError("Unsupported optimization method")

    return history


def animate_multiple_arms(histories, L1, L2, L3, target):
    """
    Create an animation showing multiple robot arms (with different optimization methods) reaching the target.
    """
    x_t, y_t = target
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        ax.clear()
        colors = ['r', 'g', 'b']  # Different colors for each arm
        labels = ['Gradient Descent', 'Momentum', 'Nesterov']

        for i, history in enumerate(histories):
            theta1, theta2, theta3 = history[frame]
            x1, y1, x2, y2, x3, y3 = forward_kinematics(theta1, theta2, theta3, L1, L2, L3)

            # Plot the arm
            ax.plot([0, x1, x2, x3], [0, y1, y2, y3], 'o-', label=labels[i], color=colors[i], markersize=8)

        ax.plot(x_t, y_t, 'kx', markersize=10, label='Target')  # Target point

        ax.set_xlim(-L1 - L2 - L3 - 1, L1 + L2 + L3 + 1)
        ax.set_ylim(-L1 - L2 - L3 - 1, L1 + L2 + L3 + 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Iteration {frame + 1} - Robot Arms')
        ax.legend()
        ax.grid(True)

    ani = animation.FuncAnimation(fig, update, frames=max(len(h) for h in histories), repeat=False, interval=100)
    return ani


# Step 1: Define the arm geometry and target
L1, L2, L3 = 1.5, 0.7, 0.5  # Lengths of the robot segments
x_t, y_t = 1, 0.25  # Target point

# Initial angles
theta1 = np.pi / 6
theta2 = -np.pi / 4
theta3 = np.pi / 8
x_init = np.array([theta1, theta2, theta3])

# Step 2: Optimize using different methods
history_gd = wrap_and_optimize(L1, L2, L3, x_t, y_t, t=0.1, x_init=x_init, method='gd')
history_gdm = wrap_and_optimize(L1, L2, L3, x_t, y_t, t=0.1, x_init=x_init, method='gdm')
history_nagd = wrap_and_optimize(L1, L2, L3, x_t, y_t, t=0.1, x_init=x_init, method='nagd')

# Step 3: Animate the arms
histories = [history_gd, history_gdm, history_nagd]
ani = animate_multiple_arms(histories, L1, L2, L3, target=(x_t, y_t))

# Step 4: Save the animation as a gif
ani.save('./robot_arms_optimization.gif', writer='pillow', fps=10)
