# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:29:31 2024

@author: csdro
"""

import pickle
import numpy as np
import random
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


def animate_spline_multiple_targets(history, L1, L2, L3, targets, annotate=True):
    """
    Create an animation showing the robot arm's spline as it tries to reach multiple targets.
    
    Args:
        history: List of tuples (theta1, theta2, theta3, alpha) at each iteration.
        L1, L2, L3: Lengths of the robot arm segments.
        targets: List of target points [(x_t1, y_t1), (x_t2, y_t2), ...].
        annotate: Shows points in gif.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Total frames = history length * number of targets
    total_frames = len(history) * len(targets)

    def update(frame):
        ax.clear()

        # Get the target for this frame
        target_index = frame // len(history)  # Cycle through targets
        x_t, y_t = targets[target_index % len(targets)]  # Get current target

        # Get current joint angles
        theta1, theta2, theta3 = history[frame % len(history)]  # Wrap frame within history length

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
        fig, update, frames=total_frames, repeat=False, interval=100
    )

    return ani

# Step 1: Define the arm geometry and target
L1, L2, L3 = 1.5, 0.7, 0.5  # Lengths of the robot segments
random_targets = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(5)]  # Random targets

# Initial angles
theta1 = np.pi / 6
theta2 = -np.pi / 4
theta3 = np.pi / 8
x_init = np.array([theta1, theta2, theta3])

# Step 2: Optimize for each target
history_all_targets = []
for target in random_targets:
    x_t, y_t = target
    history = wrap_and_optimize(L1, L2, L3, x_t, y_t, t=0.1, x_init=x_init)
    history_all_targets.append(history)

# Flatten history for all targets
history_flat = [item for sublist in history_all_targets for item in sublist]

# Step 3: Animate for all targets
ani = animate_spline_multiple_targets(history_flat, L1, L2, L3, targets=random_targets)

# Step 4: Save the animation as a gif
ani.save('./roub1.gif', writer='pillow', fps=10)
