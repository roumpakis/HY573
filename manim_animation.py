from manim import *
import pickle
import numpy as np

class RobotArmAnimation(Scene):
    def construct(self):
        # Load the history from the pickle file (adjust path as needed)
        with open("history_polar_2.pkl", "rb") as f:
            history = pickle.load(f)

        # Parameters
        L1, L2, L3 = 1.5, 0.7, 0.5  # Lengths of the segments
        x_t, y_t = 1, 0.25  # Target point

        # Target point
        # Target point
        target = Dot(point=[x_t, y_t, 0], color=RED)

        # Updated target label: smaller font size and positioned more to the right
        target_label = Text(f"Target ({x_t:.2f}, {y_t:.2f})", font_size=18).next_to(target, RIGHT, buff=0.3)


        # Base point
        base_dot = Dot(point=[0, 0, 0], color=WHITE)
        base_image = ImageMobject("b5.png")  # Ensure the path is correct
        base_image.scale(0.3)  # Scale the image to an appropriate size

        # Add target, base dot, and image to the scene
        self.add(target, target_label, base_dot, base_image)

        # Create lines for the arm
        arm_segment_1 = Line(start=[0, 0, 0], end=[0, 0, 0], color=BLUE, stroke_width=6)
        arm_segment_2 = Line(start=[0, 0, 0], end=[0, 0, 0], color=GREEN, stroke_width=6)
        arm_segment_3 = Line(start=[0, 0, 0], end=[0, 0, 0], color=YELLOW, stroke_width=6)
        end_effector = Dot(point=[0, 0, 0], color=YELLOW)

        # Add the arm to the scene
        self.add(arm_segment_1, arm_segment_2, arm_segment_3, end_effector)

        # Create a text object to display the iteration number
        iteration_text = Text("Iteration: 0", font_size=24).to_corner(UR)

        # Add the iteration text to the scene
        self.add(iteration_text)

        # Create text objects for current (x, y) position and target position
        current_position_text = Text("Current (0.00, 0.00)", font_size=24).to_edge(LEFT)
        self.add(current_position_text)  # Add current position text to the scene

        # Animate the arm through the history
        for i, (theta1, theta2, theta3) in enumerate(history):
            # Calculate joint and end-effector positions for 3 DOF arm
            x1 = L1 * np.cos(theta1)
            y1 = L1 * np.sin(theta1)
            x2 = x1 + L2 * np.cos(theta1 + theta2)
            y2 = y1 + L2 * np.sin(theta1 + theta2)
            x3 = x2 + L3 * np.cos(theta1 + theta2 + theta3)
            y3 = y2 + L3 * np.sin(theta1 + theta2 + theta3)

            # Update arm positions and the current position text
            self.play(
                arm_segment_1.animate.put_start_and_end_on([0, 0, 0], [x1, y1, 0]),
                arm_segment_2.animate.put_start_and_end_on([x1, y1, 0], [x2, y2, 0]),
                arm_segment_3.animate.put_start_and_end_on([x2, y2, 0], [x3, y3, 0]),
                end_effector.animate.move_to([x3, y3, 0]),
                
                # Update the current position text with new values
                current_position_text.animate.become(Text(f"Current ({x3:.2f}, {y3:.2f})", font_size=24).to_edge(LEFT)),
                
                # Update the iteration text to show the current frame number
                iteration_text.animate.become(Text(f"Iteration: {i+1}", font_size=24).to_corner(UR)),
                run_time=0.1,
            )

        self.wait(2)
