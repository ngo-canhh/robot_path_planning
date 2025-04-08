import numpy as np
from indoor_robot_env import IndoorRobotEnv
from obstacle import StaticObstacle, DynamicObstacle
from shape import Circle, Rectangle

# --- ObstacleIdentifier ---

class ObstacleIdentifier:
    def __init__(self, env_observation_space, max_obstacles_in_observation):
        self.observation_space = env_observation_space # May not be needed directly
        self.max_obstacles = max_obstacles_in_observation
        # Constants from env
        self.obs_data_start_index = IndoorRobotEnv.OBS_ROBOT_STATE_SIZE
        self.obstacle_data_size = IndoorRobotEnv.OBS_OBSTACLE_DATA_SIZE

    def identify(self, observation: np.ndarray) -> list:
        """
        Identify obstacles from the observation array.

        Args:
            observation: The observation numpy array from the environment.

        Returns:
            List of perceived Obstacle objects (StaticObstacle or DynamicObstacle).
        """
        perceived_obstacles = []

        for i in range(self.max_obstacles):
            base_idx = self.obs_data_start_index + i * self.obstacle_data_size
            # Check if index is within bounds
            if base_idx + self.obstacle_data_size > len(observation):
                 break # Stop parsing if we go beyond observation length

            # Extract data according to the new format
            obs_x = observation[base_idx]
            obs_y = observation[base_idx + 1]
            shape_type_enum = int(round(observation[base_idx + 2])) # Round float to int
            param1 = observation[base_idx + 3]
            param2 = observation[base_idx + 4]
            param3 = observation[base_idx + 5]
            is_dynamic_flag = observation[base_idx + 6]
            vel_x = observation[base_idx + 7]
            vel_y = observation[base_idx + 8]

            # Check if it's valid obstacle data (e.g., non-zero position/params indicate not padding)
            # A robust check might be needed based on how padding is done (e.g., shape_type == -1 for pad?)
            # Using param1 > 0 as a proxy (assumes radius/width are always > 0 for valid obstacles)
            if param1 > 1e-3:
                shape = None
                try:
                    if shape_type_enum == Circle.SHAPE_TYPE_ENUM: # Circle
                        radius = param1
                        if radius > 1e-3: shape = Circle(radius)
                    elif shape_type_enum == Rectangle.SHAPE_TYPE_ENUM: # Rectangle
                        width, height, angle = param1, param2, param3
                        if width > 1e-3 and height > 1e-3: shape = Rectangle(width, height, angle)
                    # Add elif for Polygon etc. if implemented
                except ValueError as e:
                    # print(f"Warning: Invalid shape parameters in observation for obstacle {i}: {e}")
                    shape = None # Could not create shape

                if shape is None:
                    # print(f"Warning: Could not determine shape for observed obstacle {i}, skipping.")
                    continue # Skip if shape couldn't be created

                is_dynamic = is_dynamic_flag > 0.5
                if is_dynamic:
                    velocity = np.sqrt(vel_x**2 + vel_y**2)
                    direction = np.array([vel_x, vel_y])
                    if velocity > 1e-6:
                        direction = direction / velocity
                    else:
                        direction = np.array([0.0, 0.0])
                    # Create DynamicObstacle
                    perceived_obstacles.append(DynamicObstacle(obs_x, obs_y, shape, velocity, direction))
                else:
                    # Create StaticObstacle
                    perceived_obstacles.append(StaticObstacle(obs_x, obs_y, shape))

        return perceived_obstacles