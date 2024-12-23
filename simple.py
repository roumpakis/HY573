import pickle
import numpy as np
import optimization_methods as tools
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def forward_kinematics(theta1, theta2, L1, L2):
    """
    Compute the positions of the joint and end-effector given angles and lengths.
    """
    # Joint position (x1, y1)
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    
    # End-effector position (x2, y2)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    
    return x1, y1, x2, y2

def linear_spline_cartesian(theta1, L1, theta2, L2):
    """
    Given angles (theta1, theta2) and lengths (L1, L2), return the linear spline 
    defined by the robot arm, handling vertical line edge cases.
    """
    # Joint position (x1, y1)
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    
    # End-effector position (x2, y2)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    
    # First segment spline coefficients (p1(x) = a_0 + a_1 * x)
    if np.abs(x1) > 1e-10:  # Avoid division by zero
        a1 = y1 / x1
        a0 = 0  # Since the base (x0, y0) = (0, 0)
    else:  # Handle vertical line
        a1 = None
        a0 = y1  # Vertical line means no slope, fixed y-intercept
        
    # Second segment spline coefficients (p2(x) = b_0 + b_1 * (x - x1))
    if np.abs(x2 - x1) > 1e-10:  # Avoid division by zero
        b1 = (y2 - y1) / (x2 - x1)
        b0 = y1 - b1 * x1
    else:  # Handle vertical line
        b1 = None
        b0 = y2  # Vertical line means no slope, fixed y-intercept
    
    return (a0, a1), (b0, b1)


def plot_spline(theta1, L1, theta2, L2):
    """
    Plot the linear spline defined by the robot arm given angles and lengths.
    """
    # Compute joint and end-effector positions
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    
    # Define the arm segments
    x_coords = [0, x1, x2]
    y_coords = [0, y1, y2]
    
    # Plot the arm and splines
    plt.figure(figsize=(6, 6))
    plt.plot(x_coords, y_coords, 'o-', label='Robot Arm', markersize=8)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)  # x-axis reference
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)  # y-axis reference
    
    # Annotate points
    plt.text(0, 0, 'Base (0,0)', fontsize=9, ha='right')
    plt.text(x1, y1, f'Joint ({x1:.2f},{y1:.2f})', fontsize=9, ha='right')
    plt.text(x2, y2, f'End-Effector ({x2:.2f},{y2:.2f})', fontsize=9, ha='right')
    
    # Set plot limits and labels
    plt.xlim(-L1 - L2 - 1, L1 + L2 + 1)
    plt.ylim(-L1 - L2 - 1, L1 + L2 + 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Spline of the Robot Arm')
    plt.legend()
    plt.grid(True)
    plt.show()


def objective_function_polar(thetas, L1, L2, x_t, y_t):
    """
    Calculate the squared distance between the end-effector and the target point.
    """
    theta1, theta2 = thetas[0], thetas[1]
    _, _, x2, y2 = forward_kinematics(theta1, theta2, L1, L2)
    return (x2 - x_t)**2 + (y2 - y_t)**2


def gradients_polar(thetas, L1, L2, x_t, y_t):

    """
    Compute the gradients of the squared distance with respect to theta1 and theta2.
    """
    theta1, theta2 = thetas[0], thetas[1]
    # Forward kinematics to compute positions
    x1, y1, x2, y2 = forward_kinematics(theta1, theta2, L1, L2)

    

    # Partial derivatives of x2 and y2 with respect to theta1 and theta2
    dx2_dtheta1 = -L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2)
    dy2_dtheta1 = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    dx2_dtheta2 = -L2 * np.sin(theta1 + theta2)
    dy2_dtheta2 = L2 * np.cos(theta1 + theta2)

    
    # Gradients of the objective function with respect to theta1 and theta2
    grad_theta1 = 2 * (x2 - x_t) * dx2_dtheta1 + 2 * (y2 - y_t) * dy2_dtheta1
    grad_theta2 = 2 * (x2 - x_t) * dx2_dtheta2 + 2 * (y2 - y_t) * dy2_dtheta2

    return np.array([grad_theta1, grad_theta2])

def gradient_descent_polar(theta1, theta2, L1, L2, x_t, y_t, alpha, iterations):
    """
    Perform gradient descent to optimize the angles theta1 and theta2.
    """
    history = []  # To store the angles and distance at each step
    
    for i in range(iterations):
        # Compute gradients
        grad_theta1, grad_theta2 = gradients_polar(theta1, theta2,
                                                   L1, L2, x_t, y_t)
        
        # Update angles
        theta1 -= alpha * grad_theta1
        theta2 -= alpha * grad_theta2
        
        # Compute current distance
        current_distance = objective_function_polar(theta1, theta2,
                                                    L1, L2, x_t, y_t)
        history.append((theta1, theta2, current_distance, alpha))
    
    return theta1, theta2, history

def wrap_and_optimize(L1, L2, x_t, y_t, t, x_init, lamda=None,
                                     max_iter=4000, tol=1e-6, stop_criterion=0):
    """
    Optimize the robot arm angles using the existing nesterov_momentum function.
    """
    # Wrap the objective and gradient for nesterov_momentum
    def func(thetas):
        return  objective_function_polar(thetas, L1, L2, x_t, y_t)

    def dfunc(thetas):
        return gradients_polar(thetas, L1, L2, x_t, y_t)

    # Use nesterov_momentum for optimization
    history = tools.nesterov_momentum(func, dfunc, t, lamda=lamda, x_init=x_init,
                                max_iter=max_iter, tol=tol, stop_criterion=stop_criterion)
    return history

def animate_spline(history, L1, L2, target, anotate=True):

    """
    Create an animation showing the robot arm's spline as it tries to reach the target.
    
    Args:
        history: List of tuples (theta1, theta2, distance) at each iteration.
        L1, L2: Lengths of the robot arm segments.
        target: Target point (x_t, y_t).
        anotate: Shows points in gif.
    """

    x_t, y_t = target

    fig, ax = plt.subplots(figsize=(6, 6))



    def update(frame):
        ax.clear()
        theta1, theta2= history[frame]

        # Compute joint and end-effector positions
        x1 = L1 * np.cos(theta1)
        y1 = L1 * np.sin(theta1)
        x2 = x1 + L2 * np.cos(theta1 + theta2)
        y2 = y1 + L2 * np.sin(theta1 + theta2)



        # Define the arm segments
        x_coords = [0, x1, x2]
        y_coords = [0, y1, y2]



        # Plot the arm and splines
        ax.plot(x_coords, y_coords, 'o-', label='Robot Arm', markersize=8)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)  # x-axis reference
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)  # y-axis reference
        ax.plot(x_t, y_t, 'rx', markersize=10, label='Target')  # Target point



        # Annotate points
        if anotate:
            ax.text(0, 0, 'Base (0,0)', fontsize=8, ha='right')
    
            ax.text(x1, y1, f'Joint ({x1:.2f},{y1:.2f})',
                    fontsize=8, ha='right')
    
            ax.text(x2, y2, f'End-Effector ({x2:.2f},{y2:.2f})',
                    fontsize=8, ha='right')



        # Set plot limits and labels

        ax.set_xlim(-0.25, L1 + L2 + 0.5)
        ax.set_ylim(-0.25, L1 + L2 + 0.5)
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
L1 = 1.5  # Length of the first segment
L2 = .7  # Length of the second segment
x_t, y_t = 1, 0.25  # Target point




# Initial angles 
theta1 = np.pi / 6  
theta2 = -np.pi / 4  
x_init=np.array([theta1, theta2])
# Optimize
history_polar = wrap_and_optimize(
     L1, L2, x_t, y_t, t=0.1, x_init=x_init
)

with open("history_polar.pkl", "wb") as f:
    pickle.dump(history_polar, f)
    
## ANIMATION
ani = animate_spline(history_polar, L1, L2, target=(x_t, y_t))

## Save the animation as a gif
ani.save('./simle_robot_arm_animation.gif', writer='pillow', fps=10)