import numpy as np
from components.obstacle import StaticObstacle, DynamicObstacle
from components.shape import Circle, Rectangle

OBS_ROBOT_STATE_SIZE = 5  # [x, y, angle, goal_x, goal_y]
OBS_OBSTACLE_DATA_SIZE = 9  # [x, y, shape_type, p1, p2, p3, is_dynamic, vx, vy]

class ObstacleIdentifier:
    def __init__(self, max_obstacles_in_observation: int):
        self.max_obstacles = max_obstacles_in_observation
        self.obs_data_start_index = OBS_ROBOT_STATE_SIZE
        self.obstacle_data_size = OBS_OBSTACLE_DATA_SIZE

    def identify(self, observation: np.ndarray) -> list:
        """
        Identify obstacles from the observation array.

        Args:
            observation: Numpy array from the environment.

        Returns:
            List of dictionaries, each containing obstacle data
            {'x', 'y', 'shape', 'is_dynamic', 'velocity', 'direction'}.
        """
        perceived_obstacles_data = []
        print(f"Processing observation: {observation[:self.obs_data_start_index]} (robot state)")

        for i in range(self.max_obstacles):
            base_idx = self.obs_data_start_index + i * self.obstacle_data_size
            if base_idx + self.obstacle_data_size > len(observation):
                print(f"Stopping at obstacle {i}: Index {base_idx} exceeds observation length {len(observation)}")
                break

            # Extract data
            obs_x = observation[base_idx]
            obs_y = observation[base_idx + 1]
            shape_type_enum = int(round(observation[base_idx + 2]))
            param1 = observation[base_idx + 3]
            param2 = observation[base_idx + 4]
            param3 = observation[base_idx + 5]
            is_dynamic_flag = observation[base_idx + 6]
            vel_x = observation[base_idx + 7]
            vel_y = observation[base_idx + 8]

            print(f"Obstacle {i} raw data: x={obs_x:.1f}, y={obs_y:.1f}, shape_type={shape_type_enum}, "
                  f"params=({param1:.1f},{param2:.1f},{param3:.1f}), dynamic={is_dynamic_flag:.1f}, "
                  f"vel=({vel_x:.1f},{vel_y:.1f})")

            # Validate obstacle data
            if param1 <= 1e-3 or obs_x <= 0 or obs_y <= 0:
                print(f"Skipping obstacle {i}: Invalid data (param1={param1:.1f}, x={obs_x:.1f}, y={obs_y:.1f})")
                continue

            shape = None
            try:
                if shape_type_enum == Circle.SHAPE_TYPE_ENUM:  # Circle = 0
                    radius = param1
                    if radius > 1e-3:
                        shape = Circle(radius)
                elif shape_type_enum == Rectangle.SHAPE_TYPE_ENUM:  # Rectangle = 1
                    width, height, angle = param1, param2, param3
                    if width > 1e-3 and height > 1e-3:
                        shape = Rectangle(width, height, angle)
            except ValueError as e:
                print(f"Warning: Invalid shape parameters for obstacle {i}: {e}")
                continue

            if shape is None:
                print(f"Warning: Could not create shape for obstacle {i}, skipping")
                continue

            is_dynamic = is_dynamic_flag > 0.5
            velocity = 0.0
            direction = np.array([0.0, 0.0])

            if is_dynamic:
                velocity = np.sqrt(vel_x**2 + vel_y**2)
                direction = np.array([vel_x, vel_y])
                if velocity > 1e-6:
                    direction = direction / velocity

            obstacle_data = {
                'x': obs_x,
                'y': obs_y,
                'shape': shape,
                'is_dynamic': is_dynamic,
                'velocity': velocity,
                'direction': direction
            }
            perceived_obstacles_data.append(obstacle_data)
            print(f"Added obstacle {i}: {'Dynamic' if is_dynamic else 'Static'}, "
                  f"Shape={type(shape).__name__}, Vel={velocity:.1f}")

        return perceived_obstacles_data