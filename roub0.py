# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:29:31 2024

@author: csdro
"""

import pickle
import numpy as np
import optimization_methods as tools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import optimization_methods as tools

def forward_kinematics(theta1, theta2, theta3, L1, L2, L3):
    """
    Compute the positions of the joints and end-effector given angles and lengths.
    """
    # Joint 1 position (x1, y1)
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    
    # Joint 2 position (x2, y2)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    
    # End-effector position (x3, y3)
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

    # Partial derivatives of x3 and y3 with respect to theta1, theta2, and theta3
    dx3_dtheta1 = -L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2) - L3 * np.sin(theta1 + theta2 + theta3)
    dy3_dtheta1 = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2) + L3 * np.cos(theta1 + theta2 + theta3)
    dx3_dtheta2 = -L2 * np.sin(theta1 + theta2) - L3 * np.sin(theta1 + theta2 + theta3)
    dy3_dtheta2 = L2 * np.cos(theta1 + theta2) + L3 * np.cos(theta1 + theta2 + theta3)
    dx3_dtheta3 = -L3 * np.sin(theta1 + theta2 + theta3)
    dy3_dtheta3 = L3 * np.cos(theta1 + theta2 + theta3)

    # Gradients of the objective function
    grad_theta1 = 2 * (x3 - x_t) * dx3_dtheta1 + 2 * (y3 - y_t) * dy3_dtheta1
    grad_theta2 = 2 * (x3 - x_t) * dx3_dtheta2 + 2 * (y3 - y_t) * dy3_dtheta2
    grad_theta3 = 2 * (x3 - x_t) * dx3_dtheta3 + 2 * (y3 - y_t) * dy3_dtheta3

    return np.array([grad_theta1, grad_theta2, grad_theta3])


def wrap_and_optimize(L1, L2, L3, x_t, y_t, t, x_init, lamda=None,
                      max_iter=4000, tol=1e-6, stop_criterion=0):
    """
    Optimize the robot arm angles using the existing nesterov_momentum function.
    """
    def func(thetas):
        return objective_function_polar(thetas, L1, L2, L3, x_t, y_t)

    def dfunc(thetas):
        return gradients_polar(thetas, L1, L2, L3, x_t, y_t)

    # Use nesterov_momentum for optimization
    history = tools.nesterov_momentum(func, dfunc, t, lamda=lamda, x_init=x_init,
                                      max_iter=max_iter, tol=tol, stop_criterion=stop_criterion)
    return history


def animate_spline(history, L1, L2, L3, target, annotate=True):
    """
    Create an animation showing the robot arm's spline as it tries to reach the target.
    
    Args:
        history: List of tuples (theta1, theta2, theta3, alpha) at each iteration.
        L1, L2, L3: Lengths of the robot arm segments.
        target: Target point (x_t, y_t).
        annotate: Shows points in gif.
    """
    x_t, y_t = target

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        ax.clear()
        theta1, theta2, theta3 = history[frame]

        # Compute joint and end-effector positions
        x1, y1, x2, y2, x3, y3 = forward_kinematics(theta1, theta2, theta3, L1, L2, L3)

        # Define the arm segments
        x_coords = [0, x1, x2, x3]
        y_coords = [0, y1, y2, y3]

        # Plot the arm
        ax.plot(x_coords, y_coords, 'o-', label='Robot Arm', markersize=8)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)  # x-axis reference
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)  # y-axis reference
        ax.plot(x_t, y_t, 'rx', markersize=10, label='Target')  # Target point

        # Annotate points and learning rate
        if annotate:
            ax.text(0, 0, 'Base (0,0)', fontsize=8, ha='right')
            ax.text(x1, y1, f'Joint 1 ({x1:.2f},{y1:.2f})', fontsize=8, ha='right')
            ax.text(x2, y2, f'Joint 2 ({x2:.2f},{y2:.2f})', fontsize=8, ha='right')
            ax.text(x3, y3, f'End-Effector ({x3:.2f},{y3:.2f})', fontsize=8, ha='right')
            #ax.text(0.1, -0.2, f'Learning Rate: {alpha:.6f}', fontsize=10, color='blue')

        # Set plot limits and labels
        ax.set_xlim(-L1 - L2 - L3 - 1, L1 + L2 + L3 + 1)
        ax.set_ylim(-L1 - L2 - L3 - 1, L1 + L2 + L3 + 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Iteration {frame + 1} - Robot Arm Animation')
        ax.legend()
        ax.grid(True)

    ani = animation.FuncAnimation(
        fig, update, frames=len(history), repeat=False, interval=100
    )

    return ani


# Step 1: Define the arm geometry and target
L1, L2, L3 = 1.5, 0.7, 0.5  # Lengths of the robot segments
x_t, y_t = 1, 0.25  # Target point

# Initial angles
theta1 = np.pi / 6
theta2 = -np.pi / 4
theta3 = np.pi / 8
x_init = np.array([theta1, theta2, theta3])

# Optimize
history_polar = wrap_and_optimize(
    L1, L2, L3, x_t, y_t, t=0.1, x_init=x_init
)

# Save history for reuse
with open("history_polar.pkl", "wb") as f:
    pickle.dump(history_polar, f)

# Step 2: Animate
ani = animate_spline(history_polar, L1, L2, L3, target=(x_t, y_t))

# Save the animation as a gif
ani.save('./roub0.gif', writer='pillow', fps=10)
