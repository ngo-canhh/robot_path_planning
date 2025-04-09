import numpy as np
from components.obstacle import Obstacle, ObstacleType

# --- WaitingRule ---

class WaitingRule:
    def __init__(self, robot_radius, safety_margin=20, prediction_horizon=15, time_step=0.5):
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.prediction_horizon = prediction_horizon
        self.time_step = time_step

    def check_dynamic_collisions(self, robot_x, robot_y, robot_velocity, robot_orientation, dynamic_obstacles: list):
        """ Predicts future positions and checks collisions using obstacle shapes. """
        predicted_collisions = []
        robot_effective_radius = self.robot_radius + self.safety_margin # Use safety margin here

        for obstacle in dynamic_obstacles:
            # Ensure it's dynamic and moving (already filtered by caller usually)
            if obstacle.type != ObstacleType.DYNAMIC or obstacle.velocity < 1e-3:
                 continue

            # Predict future states
            for t_step in range(1, self.prediction_horizon + 1):
                 t = t_step * self.time_step

                 future_robot_x = robot_x + robot_velocity * np.cos(robot_orientation) * t
                 future_robot_y = robot_y + robot_velocity * np.sin(robot_orientation) * t

                 # Predict obstacle future center position (linear)
                 # TODO: Could enhance prediction using obstacle's update logic (e.g., bouncing)
                 future_obstacle_x = obstacle.x + obstacle.velocity * obstacle.direction[0] * t
                 future_obstacle_y = obstacle.y + obstacle.velocity * obstacle.direction[1] * t

                 # Check collision using the obstacle's shape method at predicted locations
                 if obstacle.shape.check_collision(future_robot_x, future_robot_y,
                                                  self.robot_radius + self.safety_margin, # Check with margin
                                                  future_obstacle_x, future_obstacle_y):
                      predicted_collisions.append((obstacle, t_step))
                      # print(f"Predicted collision with dyn obs at step {t_step}")
                      break # Found collision with this obstacle

        return predicted_collisions

    def should_wait(self, predicted_collisions):
        return len(predicted_collisions) > 0