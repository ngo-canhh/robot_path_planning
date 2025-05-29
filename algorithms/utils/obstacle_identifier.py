import numpy as np
from robot_env.indoor_robot_env import IndoorRobotEnv
from components.obstacle import StaticObstacle, DynamicObstacle
from components.shape import Circle, Rectangle, Triangle, Polygon

# --- ObstacleIdentifier ---

class ObstacleIdentifier:
    def __init__(self, max_obstacles_in_observation):
        # self.observation_space = env_observation_space # May not be needed directly
        self.max_obstacles = max_obstacles_in_observation
        # # Constants from env
        # self.obs_data_start_index = IndoorRobotEnv.OBS_ROBOT_STATE_SIZE
        # self.obstacle_data_size = IndoorRobotEnv.OBS_OBSTACLE_DATA_SIZE

    def identify(self, observation: dict) -> list:
        """
        Identify obstacles from the observation array.

        Args:
            observation: The observation numpy array from the environment.

        Returns:
            List of perceived Obstacle objects (StaticObstacle or DynamicObstacle).
        """
        perceived_obstacles = []
        sensed_obstacles = observation['sensed_obstacles']

        for sensed_obs in sensed_obstacles:
            # base_idx = self.obs_data_start_index + i * self.obstacle_data_size
            # # Check if index is within bounds
            # if base_idx + self.obstacle_data_size > len(observation):
            #      break # Stop parsing if we go beyond observation length

            # Extract data according to the new format
            # print(f"Identifying obstacle: {sensed_obs}")
            obs_x = sensed_obs['x']
            obs_y = sensed_obs['y']
            shape_type_enum = sensed_obs['shape_type']
            shape_params = sensed_obs['shape_params']
            is_dynamic_flag = sensed_obs['dynamic_flag']
            vel_x = sensed_obs['vel_x']
            vel_y = sensed_obs['vel_y']
            bounding_box = sensed_obs['bounding_box'] # tuple (x_min, y_min, x_max, y_max)

            # Check if it's valid obstacle data (e.g., non-zero position/params indicate not padding)
            # A robust check might be needed based on how padding is done (e.g., shape_type == -1 for pad?)
            # Using param1 > 0 as a proxy (assumes radius/width are always > 0 for valid obstacles)
            # print(f"Shape params: {shape_params}")
            shape = None
            try:
                if shape_type_enum == Circle.SHAPE_TYPE_ENUM: # Circle
                    radius = shape_params
                    if radius > 1e-3: shape = Circle(radius)
                elif shape_type_enum == Rectangle.SHAPE_TYPE_ENUM: # Rectangle
                    width, height, angle = shape_params
                    if width > 1e-3 and height > 1e-3: shape = Rectangle(width, height, angle)
                elif shape_type_enum == Triangle.SHAPE_TYPE_ENUM: 
                    p1_x, p1_y, p2_x, p2_y, p3_x, p3_y = shape_params
                    shape = Triangle([(p1_x, p1_y), (p2_x, p2_y), (p3_x, p3_y)])
                elif shape_type_enum == Polygon.SHAPE_TYPE_ENUM: 
                    vertices = []
                    for i in range(0, len(shape_params), 2):
                        vertices.append((shape_params[i], shape_params[i+1]))
                    shape = Polygon(vertices)
                        
            except ValueError as e:
                print(f"Warning: Invalid shape parameters in observation for obstacle {sensed_obs}: {e}")
                shape = None # Could not create shape

            if shape is None:
                print(f"Warning: Could not determine shape for observed obstacle {sensed_obs}, skipping.")
                continue # Skip if shape couldn't be created

            is_dynamic = is_dynamic_flag > 0.5
            if is_dynamic:
                velocity = np.sqrt(vel_x**2 + vel_y**2)
                direction = np.array([vel_x, vel_y])
                if velocity > 1e-6:
                    direction = direction / velocity
                else:
                    direction = np.array([0.0, 0.0])

                if bounding_box is not None:
                    # Ensure bounding box is a tuple of floats
                    bounding_box = tuple(map(float, bounding_box))
                # Create DynamicObstacle
                perceived_obstacles.append(DynamicObstacle(obs_x, obs_y, shape, velocity, direction, bounding_box))
            else:
                # Create StaticObstacle
                perceived_obstacles.append(StaticObstacle(obs_x, obs_y, shape))

        return perceived_obstacles