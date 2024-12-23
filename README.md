
# Robot Arm Optimization and Animation

This Python project demonstrates a robotic arm simulation and optimization using various gradient-based methods. It includes functions for kinematics, optimization, and visualization of the robot arm as it moves towards a target.

## Features

- **Forward Kinematics**: Computes the positions of the robot arm's joint and end-effector based on angles and lengths.
- **Linear Splines**: Defines the arm's segments using linear spline interpolation.
- **Gradient Descent Optimization**:
  - Standard gradient descent for optimizing arm angles.
  - Nesterov Momentum optimization for accelerated convergence.
- **Animation**: Visualizes the robot arm's movement and convergence towards a target.
- **Exportable History**: Saves the optimization history for reproducibility or further analysis.

---

## Dependencies

The project uses the following Python libraries:
- `numpy`: Numerical computations.
- `matplotlib`: Visualization and animation.
- `pickle`: Saving and loading optimization history.
- `optimization_methods`: Custom gradient-based methods (e.g., Nesterov Momentum).

Install these dependencies using:
```bash
pip install numpy matplotlib
```

---

## File Structure

### Main Functions

1. **Kinematics**:
   - `forward_kinematics`: Computes positions of the joint and end-effector.
   - `linear_spline_cartesian`: Computes linear splines for arm segments.

2. **Optimization**:
   - `gradient_descent_polar`: Performs gradient descent to optimize arm angles.
   - `wrap_and_optimize`: Uses Nesterov Momentum for optimization.

3. **Visualization**:
   - `plot_spline`: Plots the robot arm's spline at a given state.
   - `animate_spline`: Creates an animation showing the optimization process.

4. **Objective and Gradients**:
   - `objective_function_polar`: Calculates the squared distance to the target.
   - `gradients_polar`: Computes gradients with respect to angles.

---

## How to Use

### 1. Run Optimization
Modify the target point and initial angles as needed:
```python
# Define arm lengths and target
L1 = 1.5  # Length of the first segment
L2 = 0.7  # Length of the second segment
x_t, y_t = 1, 0.25  # Target point

# Initial angles
theta1 = np.pi / 6
theta2 = -np.pi / 4
x_init = np.array([theta1, theta2])

# Optimize
history_polar = wrap_and_optimize(
    L1, L2, x_t, y_t, t=0.1, x_init=x_init
)
```

### 2. Save and Load History
Save the optimization history:
```python
with open("history_polar.pkl", "wb") as f:
    pickle.dump(history_polar, f)
```

Load it later:
```python
with open("history_polar.pkl", "rb") as f:
    history_polar = pickle.load(f)
```

### 3. Animate the Robot Arm
Generate an animation of the optimization:
```python
ani = animate_spline(history_polar, L1, L2, target=(x_t, y_t))
ani.save('./robot_arm_animation.gif', writer='pillow', fps=10)
```

---

## Example Workflow

```python
# Step 1: Define the arm geometry and target
L1 = 1.5
L2 = 0.7
x_t, y_t = 1, 0.25

# Step 2: Initial angles
theta1 = np.pi / 6
theta2 = -np.pi / 4
x_init = np.array([theta1, theta2])

# Step 3: Optimize
history_polar = wrap_and_optimize(
    L1, L2, x_t, y_t, t=0.1, x_init=x_init
)

# Step 4: Animate
ani = animate_spline(history_polar, L1, L2, target=(x_t, y_t))
ani.save('./robot_arm_animation.gif', writer='pillow', fps=10)
```

---

## Outputs

1. **Optimization History**: Saved as `history_polar.pkl`.
2. **Animation**: A `.gif` file visualizing the optimization process.

---

## Notes

- Ensure that `optimization_methods` includes the `nesterov_momentum` function, or update the import if required.
- Adjust step size `t` and momentum parameter `lamda` to fine-tune optimization performance.

---

