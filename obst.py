import pickle
import numpy as np
import optimization_methods as tools
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def forward_kinematics(theta1, theta2, L1, L2):
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    return x1, y1, x2, y2

def distance_from_point_to_line(x1, y1, x2, y2, x_o, y_o):
    # Calculate the perpendicular distance from a point (x_o, y_o) to the line segment (x1, y1) -> (x2, y2)
    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if line_length == 0:
        return np.sqrt((x_o - x1)**2 + (y_o - y1)**2)
    
    # Calculate the projection of (x_o, y_o) onto the line segment
    t = ((x_o - x1) * (x2 - x1) + (y_o - y1) * (y2 - y1)) / (line_length**2)
    t = max(0, min(1, t))  # Clamp t to the segment
    closest_point_x = x1 + t * (x2 - x1)
    closest_point_y = y1 + t * (y2 - y1)
    
    return np.sqrt((x_o - closest_point_x)**2 + (y_o - closest_point_y)**2)

def obstacle_penalty(theta1, theta2, L1, L2, x_o, y_o, r_o, e):
    # Calculate arm segments (base to joint, joint to end-effector)
    x1, y1, x2, y2 = forward_kinematics(theta1, theta2, L1, L2)
    
    # Check the distance of each arm segment from the obstacle
    distance_to_base = distance_from_point_to_line(0, 0, x1, y1, x_o, y_o)
    distance_to_joint = distance_from_point_to_line(x1, y1, x2, y2, x_o, y_o)
    
    penalty = 0
    if distance_to_base < e:
        penalty += (e - distance_to_base)  # Penalize if too close
    if distance_to_joint < e:
        penalty += (e - distance_to_joint)  # Penalize if too close
    
    return penalty

def objective_with_obstacle(thetas, L1, L2, x_t, y_t, x_o, y_o, r_o, e, lambda_penalty):
    penalty = obstacle_penalty(thetas[0], thetas[1], L1, L2, x_o, y_o, r_o, e)
    return objective_function_polar(thetas, L1, L2, x_t, y_t) + lambda_penalty * penalty

def objective_function_polar(thetas, L1, L2, x_t, y_t):
    theta1, theta2 = thetas[0], thetas[1]
    _, _, x2, y2 = forward_kinematics(theta1, theta2, L1, L2)
    return (x2 - x_t)**2 + (y2 - y_t)**2

def gradients_with_obstacle(thetas, L1, L2, x_t, y_t, x_o, y_o, r_o, e, lambda_penalty):
    theta1, theta2 = thetas[0], thetas[1]
    x1, y1, x2, y2 = forward_kinematics(theta1, theta2, L1, L2)

    dx2_dtheta1 = -L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2)
    dy2_dtheta1 = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    dx2_dtheta2 = -L2 * np.sin(theta1 + theta2)
    dy2_dtheta2 = L2 * np.cos(theta1 + theta2)

    grad_theta1 = 2 * (x2 - x_t) * dx2_dtheta1 + 2 * (y2 - y_t) * dy2_dtheta1
    grad_theta2 = 2 * (x2 - x_t) * dx2_dtheta2 + 2 * (y2 - y_t) * dy2_dtheta2

    # Add penalty gradient (not implemented for simplicity)
    return np.array([grad_theta1, grad_theta2])

def wrap_and_optimize_with_obstacle(L1, L2, x_t, y_t, t, x_init, x_o, y_o, r_o, e, lambda_penalty,
                                     max_iter=4000, tol=1e-6, stop_criterion=0):
    def func(thetas):
        return objective_with_obstacle(thetas, L1, L2, x_t, y_t, x_o, y_o, r_o, e, lambda_penalty)

    def dfunc(thetas):
        return gradients_with_obstacle(thetas, L1, L2, x_t, y_t, x_o, y_o, r_o, e, lambda_penalty)

    history = tools.nesterov_momentum(func, dfunc, t, lamda=None, x_init=x_init,
                                      max_iter=max_iter, tol=tol, stop_criterion=stop_criterion)
    return history

# Initial Check to Ensure No Intersection at the Initial Position
def initial_position_check(L1, L2, x_o, y_o, r_o, e, theta1, theta2):
    x1, y1, x2, y2 = forward_kinematics(theta1, theta2, L1, L2)
    
    # Check distance of the arm from the obstacle
    distance_to_base = distance_from_point_to_line(0, 0, x1, y1, x_o, y_o)
    distance_to_joint = distance_from_point_to_line(x1, y1, x2, y2, x_o, y_o)
    
    if distance_to_base < e or distance_to_joint < e:
        return False  # Initial position is too close to the obstacle
    return True

def animate_spline(history, L1, L2, target, x_o, y_o, r_o, e, annotate=True):
    x_t, y_t = target
    fig, ax = plt.subplots(figsize=(6, 6))

    def plot_obstacle(ax):
        circle = plt.Circle((x_o, y_o), r_o + e, color='red', alpha=0.3, label='Obstacle')
        ax.add_artist(circle)

    def update(frame):
        ax.clear()
        theta1, theta2 = history[frame]

        x1 = L1 * np.cos(theta1)
        y1 = L1 * np.sin(theta1)
        x2 = x1 + L2 * np.cos(theta1 + theta2)
        y2 = y1 + L2 * np.sin(theta1 + theta2)

        x_coords = [0, x1, x2]
        y_coords = [0, y1, y2]

        ax.plot(x_coords, y_coords, 'o-', label='Robot Arm', markersize=8)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.plot(x_t, y_t, 'rx', markersize=10, label='Target')

        plot_obstacle(ax)

        if annotate:
            ax.text(0, 0, 'Base (0,0)', fontsize=8, ha='right')
            ax.text(x1, y1, f'Joint ({x1:.2f},{y1:.2f})', fontsize=8, ha='right')
            ax.text(x2, y2, f'End-Effector ({x2:.2f},{y2:.2f})', fontsize=8, ha='right')

        ax.set_xlim(-0.25, L1 + L2 + 0.5)
        ax.set_ylim(-0.25, L1 + L2 + 0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Iteration {frame + 1} - Robot Arm Animation')
        ax.legend()
        ax.grid(True)

    ani = animation.FuncAnimation(fig, update, frames=len(history), repeat=False, interval=100)
    return ani

# Parameters
L1 = 1.5
L2 = 0.7
x_t, y_t = 1, 0.25
x_o, y_o = 0.5, 0.5  # Obstacle center
r_o = 0.05
e = 0.1
lambda_penalty = 100

# Initial angles
theta1 = np.pi / 2  # Changed initial position
theta2 = -0
x_init = np.array([theta1, theta2])

# Check initial position validity
if not initial_position_check(L1, L2, x_o, y_o, r_o, e, theta1, theta2):
    print("Initial position is too close to the obstacle. Adjusting starting angles.")
    # Adjust angles or re-initialize as necessary
else:
    # Optimize
    history_polar = wrap_and_optimize_with_obstacle(
        L1, L2, x_t, y_t, t=0.1, x_init=x_init, x_o=x_o, y_o=y_o, r_o=r_o, e=e, lambda_penalty=lambda_penalty
    )

    # Save results
    with open("history_polar_with_obstacle.pkl", "wb") as f:
        pickle.dump(history_polar, f)

    # Animation
    ani = animate_spline(history_polar, L1, L2, target=(x_t, y_t), x_o=x_o, y_o=y_o, r_o=r_o, e=e)
    ani.save('./robot_arm_with_obstacle.gif', writer='pillow', fps=10)
