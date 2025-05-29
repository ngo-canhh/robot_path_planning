import math
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gymnasium as gym
from gymnasium import spaces
import random
from components.shape import Shape, Circle, Rectangle, Triangle, Polygon
from components.obstacle import Obstacle, StaticObstacle, DynamicObstacle, ObstacleType
import copy
from collections.abc import Iterable
from utils.oracle_path_planner import OraclePathPlanner  # Import the Oracle Path Planner



# --- IndoorRobotEnv ---

class IndoorRobotEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    # # Define constants for observation space structure
    # OBS_ROBOT_STATE_SIZE = 5 # x, y, orientation, gx, gy
    # OBS_OBSTACLE_DATA_SIZE = 9 # x, y, shape_type, p1, p2, p3, is_dynamic, vx, vy

    def __init__(
            self, 
            width=500, 
            height=500, 
            start=None,
            goal=None,
            robot_radius=10, 
            max_steps=1000, 
            sensor_range=150, 
            render_mode='rgb_array', 
            obs_chance_dynamic=0.3,
            config_path=None,
            metrics_callback=None,
            random_obstacles=False
            ):
        super(IndoorRobotEnv, self).__init__()

        self.width = width
        self.height = height
        self.robot_radius = robot_radius
        self.max_steps = max_steps
        self.current_step = 0
        self.sensor_range = sensor_range
        self.render_mode = render_mode
        self.dt = 0.5
        self.obs_chance_dynamic = obs_chance_dynamic # Probability of dynamic obstacle
        self.min_start_goal_dist = 350
        self.metrics_callback = metrics_callback

        self.robot_x = None
        self.robot_y = None
        self.robot_orientation = None
        self.robot_velocity = 0
        self.prev_robot_velocity = 0  # To calculate acceleration
        self.start = start
        self.goal = goal
        self.start_x = None
        self.start_y = None
        self.goal_x = None
        self.goal_y = None

        # Metrics tracking - simplified according to the paper's formulas
        self.metrics = {
            'path_length': 0.0,        # Length(P) = Sum of Euclidean distances between consecutive points
            'path_angles': [],         # Stores angles between consecutive path segments
            'path_smoothness': 0.0,    # Smoothness(P) = (1/N) * Sum of angles between segments
            'success': 0,              # Binary: 1 for success, 0 for failure
            'oracle_shortest_path': 0.0,  # l = Oracle shortest path length
            'oracle_smoothest_path': 0.0,  # s = Oracle smoothest path smoothness
            'spl': 0.0,                # SPL(P) = S * (l / max(l, Length(P)))
            'sps': 0.0,                # SPS(P) = S * (s / max(s, Smoothness(P)))
        }
        
        # Default values for oracle metrics - these should be replaced with actual computations
        # or passed from external path planning algorithms
        self.oracle_shortest_path_length = 0.0
        self.oracle_smoothest_path_smoothness = 0.0

        self.vanilla_obstacles = [] # List of Obstacle objects load from config
        self.obstacles = [] # List of Obstacle objects (ground truth)

        # Observation space: [robot_state..., sensed_obstacle_info...]
        self.max_obstacles_in_observation = 100 # Max obstacles reported
        # obs_len = self.OBS_ROBOT_STATE_SIZE + self.max_obstacles_in_observation * self.OBS_OBSTACLE_DATA_SIZE

        # # Define bounds - Use large enough bounds for shape parameters and velocities
        # obs_low = np.full(obs_len, -np.inf, dtype=np.float32)
        # obs_high = np.full(obs_len, np.inf, dtype=np.float32)

        # # Set specific bounds for known parts
        # obs_low[0:2] = 0.0                  # robot x, y min
        # obs_high[0:2] = [width, height]     # robot x, y max
        # obs_low[2] = -np.pi                 # robot orientation min
        # obs_high[2] = np.pi                 # robot orientation max
        # obs_low[3:5] = 0.0                  # goal x, y min
        # obs_high[3:5] = [width, height]     # goal x, y max

        # # Set bounds for obstacle data within the observation loop if needed
        # # For simplicity, we use inf/ -inf for now, padding will use 0
        # for i in range(self.max_obstacles_in_observation):
        #      base = self.OBS_ROBOT_STATE_SIZE + i * self.OBS_OBSTACLE_DATA_SIZE
        #      obs_low[base:base+2] = 0.0             # obs x, y min
        #      obs_high[base:base+2] = [width, height] # obs x, y max
        #      obs_low[base+2] = 0                    # obs shape type min
        #      obs_high[base+2] = 10                  # obs shape type max (allow expansion)
        #      # Params p1, p2, p3 depend on shape, use inf bounds
        #      obs_low[base+6] = 0                    # is_dynamic flag min
        #      obs_high[base+6] = 1                   # is_dynamic flag max
        #      # Velocities use inf bounds

        # self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        if config_path is not None:
            self._load_config(config_path=config_path)

        self.random_obstacles = random_obstacles

        # Action space: [velocity, steering_angle] - Limit velocity slightly
        self.action_space = spaces.Box(low=np.array([0, -np.pi * 1]),
                                      high=np.array([0.02 * self.width, np.pi * 1]), # Max vel 2% of width
                                      dtype=np.float32)

        # Visualization elements
        self.fig = None
        self.ax = None
        self.robot_patch = None
        self.goal_patch = None
        self.goal_text = None # Added handle for goal text
        self.obstacle_patches = [] # Now stores patches generated by obstacles
        self.path = []
        self.ray_lines = []
        self.rrt_tree_lines = []
        self.planned_path_line = None
        self.direction_arrow = None
        self.path_line = None
        self.oracle_path_line = None
        self.sensor_circle = None
        
        # Oracle path storage
        self.oracle_raw_path = None
        self.oracle_smoothed_path = None

        # Reset environment
        self.reset()

    def _load_config(self, config_path):
        """
        Load env from config path.
        Input: 
            config_path: str
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.width = config['environment']['width']
        self.height = config['environment']['height']
        obstacles_data = config['obstacles']
        for data in obstacles_data:
            obs_x = data['x']
            obs_y = data['y']
            shape_type_enum = data['shape_type']
            shape_params = data['shape_params']
            is_dynamic_flag = data['dynamic_flag']
            vel_x = data['vel_x']
            vel_y = data['vel_y']
            bounding_box = data['bounding_box'] # tuple (x_min, y_min, x_max, y_max)

            shape = None
            try:
                if shape_type_enum == Circle.SHAPE_TYPE_ENUM: # Circle
                    radius = shape_params[0]
                    if radius > 1e-3: shape = Circle(radius)
                elif shape_type_enum == Rectangle.SHAPE_TYPE_ENUM: # Rectangle
                    width, height, angle = shape_params
                    if width > 1e-3 and height > 1e-3: shape = Rectangle(width, height, angle)
                elif shape_type_enum == Triangle.SHAPE_TYPE_ENUM: 
                    p1_x, p1_y, p2_x, p2_y, p3_x, p3_y = shape_params
                    shape = Triangle([(p1_x, p1_y), (p2_x, p2_y), (p3_x, p3_y)])
                elif shape_type_enum == Polygon.SHAPE_TYPE_ENUM: 
                    # print(f"Polygon shape params: {shape_params}")
                    vertices = []
                    for i in range(0, len(shape_params), 2):
                        vertices.append((shape_params[i], shape_params[i+1]))
                    shape = Polygon(vertices)
                        
            except ValueError as e:
                print(f"Warning: Invalid shape parameters in observation for obstacle {data}: {e}")
                shape = None # Could not create shape

            if shape is None:
                print(f"Warning: Could not determine shape for observed obstacle {data}, skipping.")
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
                    bounding_box = tuple(bounding_box)
                # Create DynamicObstacle
                self.vanilla_obstacles.append(DynamicObstacle(obs_x, obs_y, shape, velocity, direction, bounding_box))
            else:
                # Create StaticObstacle
                self.vanilla_obstacles.append(StaticObstacle(obs_x, obs_y, shape))

    def add_obstacles(self, obstacles: Iterable[Obstacle]):
        """
        Add obstacles to the environment.
        Input:
            obstacles: Iterable of Obstacle object
        """
        self.vanilla_obstacles.extend(obstacles)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Reset metrics for new episode
        self.metrics['path_length'] = 0.0
        self.metrics['path_angles'] = []
        self.metrics['path_smoothness'] = 0.0
        self.metrics['success'] = 0
        self.metrics['spl'] = 0.0
        self.metrics['sps'] = 0.0
        self.prev_robot_velocity = 0

        margin = self.robot_radius + 10
        if self.start is None:
            self.start_x = self.np_random.uniform(margin, self.width - margin)
            self.start_y = self.np_random.uniform(margin, self.height - margin)
            self.start = (self.start_x, self.start_y)
        else:
            self.start_x, self.start_y = self.start
        
        self.robot_x, self.robot_y = self.start
        
        self.robot_orientation = 0
        self.robot_velocity = 0

        if self.goal is None:
            min_start_goal_dist = self.min_start_goal_dist
            attempts = 0
            while True:
                attempts += 1
                self.goal_x = self.np_random.uniform(margin, self.width - margin)
                self.goal_y = self.np_random.uniform(margin, self.height - margin)
                if np.sqrt((self.goal_x - self.robot_x)**2 + (self.goal_y - self.robot_y)**2) >= min_start_goal_dist:
                    break
                if attempts > 1000:
                    print(f"Warning: Could not place goal, space might be too crowded but distance start goal is so long {min_start_goal_dist}.")
                    # Fallback: place goal at a random point
                    self.goal_x = self.np_random.uniform(margin, self.width - margin)
                    self.goal_y = self.np_random.uniform(margin, self.height - margin)
                    break
            self.goal = self.goal_x, self.goal_y
        else:
            self.goal_x, self.goal_y = self.goal
        
        # Generate random obstacles or use the ones provided
        if self.random_obstacles:
            # Generate random obstacles (ground truth)
            self.obstacles = []
            num_obstacles = self.np_random.integers(5, self.max_obstacles_in_observation + 1)

            for _ in range(num_obstacles):
                attempts = 0
                while True: # Ensure obstacle placement is valid
                    attempts += 1
                    if attempts > 100:
                        print("Warning: Could not place obstacle, space might be too crowded.")
                        break

                    # Choose shape type
                    obs_x = self.np_random.uniform(margin, self.width - margin)
                    obs_y = self.np_random.uniform(margin, self.height - margin)

                    shape = Shape.create_random_shape(seed+_)

                    placement_radius = shape.get_effective_radius() # Use shape's effective radius for placement check

                    # Check minimum distance from robot start and goal (using center point and placement_radius)
                    dist_to_start = np.sqrt((obs_x - self.robot_x)**2 + (obs_y - self.robot_y)**2)
                    dist_to_goal = np.sqrt((obs_x - self.goal_x)**2 + (obs_y - self.goal_y)**2)

                    # Check minimum distance from other obstacles (center-to-center + radii)
                    # This is approximate for non-circles but prevents centroid overlap.
                    clear_of_others = True
                    for existing_obs in self.obstacles:
                        # Use placement radii for existing obstacles too
                        existing_placement_radius = existing_obs.shape.get_effective_radius()

                        dist_to_existing = np.sqrt((obs_x - existing_obs.x)**2 + (obs_y - existing_obs.y)**2)
                        # Ensure spacing between estimated boundaries + robot radius buffer
                        if dist_to_existing < placement_radius + existing_placement_radius + self.robot_radius:
                            clear_of_others = False
                            break

                    # Combine checks (using placement_radius)
                    if (dist_to_start > placement_radius + self.robot_radius + 10 and
                        dist_to_goal > placement_radius + self.robot_radius + 10 and
                        clear_of_others):
                        break # Valid placement found

                if attempts > 100: continue # Go to next obstacle if placement failed

                # Determine obstacle type (Static/Dynamic)
                if self.np_random.random() < self.obs_chance_dynamic: # 60% chance dynamic
                    obs_type = ObstacleType.DYNAMIC
                    obs_velocity = self.np_random.uniform(1.0, self.action_space.high[0] * 0.6) 
                    angle = self.np_random.uniform(0, 2*np.pi)
                    obs_direction = np.array([np.cos(angle), np.sin(angle)])
                    obstacle = DynamicObstacle(obs_x, obs_y, shape, obs_velocity, obs_direction)
                else:
                    obs_type = ObstacleType.STATIC
                    obstacle = StaticObstacle(obs_x, obs_y, shape)

                self.obstacles.append(obstacle)

            # Check if goal ended up inside a generated obstacle
            goal_margin = 1.0 # Treat goal as a point
            for obs in self.obstacles:
                if obs.check_collision(self.goal_x, self.goal_y, goal_margin): # Check goal point collision
                    print("Warning: Goal spawned inside an obstacle, attempting to move goal.")
                    # Simple fix: move goal slightly away from obstacle center
                    vec_from_obs = np.array([self.goal_x - obs.x, self.goal_y - obs.y])
                    dist = np.linalg.norm(vec_from_obs)
                    # Use a heuristic move distance (e.g., based on shape size)
                    move_dist = obs.shape.get_effective_radius()

                    if dist < 1e-6: # Goal exactly at center
                        self.goal_x += move_dist
                    else:
                        # Move along the vector from obs center to goal
                        self.goal_x += vec_from_obs[0] / dist * move_dist
                        self.goal_y += vec_from_obs[1] / dist * move_dist
                    # Clamp goal to bounds
                    self.goal_x = np.clip(self.goal_x, margin, self.width - margin)
                    self.goal_y = np.clip(self.goal_y, margin, self.height - margin)
                    # Note: This simple fix might move it into *another* obstacle. A robust fix is harder.
        else:
            self.obstacles.clear()
            for obs in self.vanilla_obstacles:
                self.obstacles.append(copy.deepcopy(obs)) # Deep copy to avoid shared references

        self.path = [(self.robot_x, self.robot_y)]
        
        # Calculate oracle path metrics using A* planner
        self._calculate_oracle_path_metrics()

        observation = self._get_observation()
        info = self._get_info()

        # Clear visualization elements
        if self.ax:
             # Remove old obstacle patches explicitly
             for patch in self.obstacle_patches:
                  if patch is not None: patch.remove()
             self.obstacle_patches = []
             # Remove other elements
             if self.robot_patch: self.robot_patch.remove()
             if self.goal_patch: self.goal_patch.remove()
             if self.goal_text: self.goal_text.remove() # Remove goal text too
             if self.direction_arrow: self.direction_arrow.remove()
             if self.path_line: self.path_line.set_data([], [])
             if self.planned_path_line: self.planned_path_line.set_data([], [])
             if hasattr(self, 'oracle_path_line') and self.oracle_path_line: self.oracle_path_line.set_data([], [])
             if self.sensor_circle: self.sensor_circle.remove()
             for line in self.ray_lines: line.remove()
             self.ray_lines = []
             for line in self.rrt_tree_lines: line.remove()
             self.rrt_tree_lines = []

             # Reset handles
             self.robot_patch = None
             self.goal_patch = None
             self.goal_text = None
             self.direction_arrow = None
             self.path_line = None
             self.planned_path_line = None
             self.sensor_circle = None

        return observation, info
        
    def _calculate_oracle_path_metrics(self):
        """
        Calculate oracle path metrics using A* path planning with grid decomposition.
        This sets the oracle_shortest_path and oracle_smoothest_path values in the metrics.
        """
        # Create Oracle Path Planner
        planner = OraclePathPlanner(self.width, self.height, self.robot_radius)
        
        # Plan path from start to goal
        start = (self.start_x, self.start_y)
        goal = (self.goal_x, self.goal_y)
        
        # Copy obstacles for planning (treating dynamic obstacles as static)
        planning_obstacles = []
        for obs in self.obstacles:
            if isinstance(obs, DynamicObstacle):
                # Convert to static obstacle
                static_obs = StaticObstacle(obs.x, obs.y, obs.shape)
                planning_obstacles.append(static_obs)
            else:
                planning_obstacles.append(obs)
        
        # Plan path
        raw_path, smoothed_path, path_length, path_smoothness = planner.plan_path(start, goal, planning_obstacles)
        
        # Store oracle paths for visualization
        self.oracle_raw_path = raw_path
        self.oracle_smoothed_path = smoothed_path
        
        # Set oracle metrics
        if smoothed_path is not None:
            self.metrics['oracle_shortest_path'] = path_length
            self.metrics['oracle_smoothest_path'] = path_smoothness
        else:
            # If no path found, use straight line distance as fallback
            self.metrics['oracle_shortest_path'] = np.sqrt((self.goal_x - self.start_x)**2 + (self.goal_y - self.start_y)**2)
            self.metrics['oracle_smoothest_path'] = 0.0  # Perfect smoothness as fallback

    def _get_observation(self):
        # Base observation: robot state and goal state
        base_obs = [self.robot_x, self.robot_y, self.robot_velocity, self.robot_orientation, self.goal_x, self.goal_y]

        # Add *sensed* obstacle information within sensor_range
        sensed_obstacles_data = []
        # count = 0
        for obstacle in self.obstacles:
            # Check distance from robot center to obstacle center
            efficient_dist = obstacle.get_efficient_distance(self.robot_x, self.robot_y)

            # Simple check: if obstacle center is within sensor range
            # A better check would consider the obstacle's extent.
            if efficient_dist <= self.sensor_range:
                # Get the 9 parameters for the observation
                obs_data = obstacle.get_observation_data()
                sensed_obstacles_data.append(obs_data)
            
            if len(sensed_obstacles_data) > self.max_obstacles_in_observation:
                break
    
        return {
            "base_observation": base_obs,
            "sensed_obstacles": sensed_obstacles_data
        }

    def _get_info(self):
        current_dist_to_goal = np.sqrt((self.robot_x - self.goal_x)**2 + (self.robot_y - self.goal_y)**2)
        return {
            "distance_to_goal": current_dist_to_goal,
            "ground_truth_obstacles": self.obstacles, # Pass the actual Obstacle objects
            "metrics": self.metrics  # Add metrics to info
        }


    def step(self, action):
        self.current_step += 1
        prev_distance = np.sqrt((self.robot_x - self.goal_x)**2 + (self.robot_y - self.goal_y)**2)

        # Store previous position for path length calculation
        prev_x, prev_y = self.robot_x, self.robot_y
        prev_velocity = self.robot_velocity

        velocity, steering_angle = action
        velocity = np.clip(velocity, self.action_space.low[0], self.action_space.high[0])
        steering_angle = np.clip(steering_angle, self.action_space.low[1], self.action_space.high[1])

        dt = self.dt # Simulation step time
        self.robot_orientation += steering_angle * dt
        self.robot_orientation = np.arctan2(np.sin(self.robot_orientation), np.cos(self.robot_orientation))

        dx = velocity * np.cos(self.robot_orientation) * dt
        dy = velocity * np.sin(self.robot_orientation) * dt
        new_x = self.robot_x + dx
        new_y = self.robot_y + dy
        self.robot_velocity = velocity

        terminated = False
        truncated = False
        reward = 0
        info = {'status': 'in_progress'}

        # Check boundary collision
        if not (self.robot_radius <= new_x <= self.width - self.robot_radius and
                self.robot_radius <= new_y <= self.height - self.robot_radius):
            reward = -50
            terminated = True
            info['status'] = 'boundary_collision'
            observation = self._get_observation()
            info.update(self._get_info())
            return observation, reward, terminated, truncated, info

        # Check obstacle collision (using obstacle's collision check method)
        collision = False
        for obstacle in self.obstacles:
            # Use the obstacle's check_collision method
            if obstacle.check_collision(new_x, new_y, self.robot_radius):
                collision = True
                colliding_obs_type = type(obstacle.shape).__name__
                break

        if collision:
            reward = -50
            terminated = True
            info['status'] = f'obstacle_collision ({colliding_obs_type})'
            observation = self._get_observation()
            info.update(self._get_info())
            return observation, reward, terminated, truncated, info

        # --- Update State if No Collision ---
        self.robot_x = new_x
        self.robot_y = new_y
        self.path.append((self.robot_x, self.robot_y))

        # Update metrics
        # Calculate path segment length - Equation (1) and (2): d(pi, pi+1)
        segment_length = np.sqrt((new_x - prev_x)**2 + (new_y - prev_y)**2)
        self.metrics['path_length'] += segment_length

        # Calculate angle for path smoothness (when we have at least 3 points)
        # Formula from Equation (3): angle(pi, pj, pk) = arccos((pi→pj · pj→pk) / (d(pi,pj) · d(pj,pk)))
        if len(self.path) >= 3:
            # Instead of using consecutive points, use a window approach to better capture sharp turns
            window_size = max(3, min(10, len(self.path) // 5))  # Adaptive window size
            
            # Take three points with window spacing
            if len(self.path) >= 2 * window_size + 1:
                pi = self.path[-2 * window_size - 1]  # Further back point
                pj = self.path[-window_size - 1]      # Middle point
                pk = self.path[-1]                    # Current point
                
                # Calculate vectors pi→pj and pj→pk
                vec_pi_pj = np.array([pj[0] - pi[0], pj[1] - pi[1]])  # pi → pj
                vec_pj_pk = np.array([pk[0] - pj[0], pk[1] - pj[1]])  # pj → pk
                
                # Calculate magnitudes (distances)
                d_pi_pj = np.sqrt(np.sum(vec_pi_pj**2))  # d(pi, pj)
                d_pj_pk = np.sqrt(np.sum(vec_pj_pk**2))  # d(pj, pk)
                
                if d_pi_pj > 1e-6 and d_pj_pk > 1e-6:  # Avoid division by zero
                    # Calculate dot product for angle
                    dot_product = np.dot(vec_pi_pj, vec_pj_pk) / (d_pi_pj * d_pj_pk)
                    dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure within valid range
                    angle = np.arccos(dot_product)
                    self.metrics['path_angles'].append(angle)
                    
                    # Update smoothness using Equation (4): smoothness(P) = (1/N) * Sum(angles)
                    if self.metrics['path_angles']:
                        # N is the number of interior points (angles calculated)
                        self.metrics['path_smoothness'] = np.mean(self.metrics['path_angles'])
        
        # Update dynamic obstacles (ground truth)
        bounds = (0, 0, self.width, self.height)
        for obstacle in self.obstacles:
             obstacle.update(dt=dt, bounds=bounds) # Pass bounds for bouncing

        # --- Calculate Reward and Termination/Truncation ---
        distance_to_goal = np.sqrt((self.robot_x - self.goal_x)**2 + (self.robot_y - self.goal_y)**2)

        goal_threshold = self.robot_radius + 5
        if distance_to_goal < goal_threshold:
            reward = 200
            terminated = True
            info['status'] = 'goal_reached'
            # Update success metric (S in the equations)
            self.metrics['success'] = 1
            
            # Calculate SPL using Equation (5): SPL(P) = S * (l / max(l, length(P)))
            if self.metrics['oracle_shortest_path'] > 0:
                self.metrics['spl'] = self.metrics['success'] * (
                    self.metrics['oracle_shortest_path'] / 
                    max(self.metrics['oracle_shortest_path'], self.metrics['path_length'])
                )
            
            # Calculate SPS using Equation (6): SPS(P) = S * (s / max(s, smoothness(P)))
            if self.metrics['path_angles']:
                # Use oracle_smoothest_path as our reference value (s)
                s = self.metrics['oracle_smoothest_path']
                self.metrics['sps'] = self.metrics['success'] * (
                    s / max(s, self.metrics['path_smoothness'])
                )
            
        else:
            reward_dist = prev_distance - distance_to_goal
            reward_time = -0.5
            reward = (reward_dist * 1.5) + reward_time

            if self.current_step >= self.max_steps:
                truncated = True
                reward -= 50
                info['status'] = 'max_steps_reached'

        # Store previous velocity for next acceleration calculation
        self.prev_robot_velocity = self.robot_velocity

        observation = self._get_observation()
        info.update(self._get_info())

        # Call metrics callback if provided
        if terminated or truncated:
            if self.metrics_callback:
                self.metrics_callback(self.metrics)

        return observation, reward, terminated, truncated, info

    def render(self, mode='human', controller_info=None):
        if mode not in self.metadata['render.modes']:
             raise ValueError(f"Unsupported render mode: {mode}")

        if self.fig is None:
             if mode == 'human':
                  plt.ion()
                  self.fig, self.ax = plt.subplots(figsize=(8, 8))
             elif mode == 'rgb_array':
                  import matplotlib
                  matplotlib.use('Agg')
                  self.fig, self.ax = plt.subplots(figsize=(8, 8))
             self.ax.set_xlim(0, self.width)
             self.ax.set_ylim(0, self.height)
             self.ax.set_aspect('equal')
             self.ax.grid(True)
             plt.title("Indoor Robot Simulation (OOP Obstacles)")
             plt.xlabel("X")
             plt.ylabel("Y")

        # Clear previous dynamic elements
        for line in self.ray_lines: line.remove()
        self.ray_lines = []
        for line in self.rrt_tree_lines: line.remove()
        self.rrt_tree_lines = []
        if self.planned_path_line:
             self.planned_path_line.remove()
             self.planned_path_line = None
        if self.direction_arrow: # Remove old arrow before drawing new one
            self.direction_arrow.remove()
            self.direction_arrow = None


        # Draw Robot
        if self.robot_patch is None:
            self.robot_patch = patches.Circle((self.robot_x, self.robot_y), self.robot_radius, fc='blue', alpha=0.8, zorder=5)
            self.ax.add_patch(self.robot_patch)
        else:
            self.robot_patch.center = (self.robot_x, self.robot_y)

        # Draw Direction Arrow
        arrow_len = self.robot_radius * 1.5
        end_x = self.robot_x + arrow_len * np.cos(self.robot_orientation)
        end_y = self.robot_y + arrow_len * np.sin(self.robot_orientation)
        # Create new arrow patch each time
        self.direction_arrow = self.ax.arrow(self.robot_x, self.robot_y,
                                            end_x - self.robot_x, end_y - self.robot_y,
                                            head_width=max(self.robot_radius * 0.4, 1),
                                            head_length=max(self.robot_radius * 0.6, 1),
                                            fc='red', ec='red', length_includes_head=False, zorder=2)

        # Draw Goal
        if self.goal_patch is None:
            self.goal_patch = patches.Circle((self.goal_x, self.goal_y), self.robot_radius * 0.8, fc='lime', alpha=0.8, ec='green', lw=2, zorder=4)
            self.ax.add_patch(self.goal_patch)
            self.goal_text = self.ax.text(self.goal_x, self.goal_y, 'G', ha='center', va='center', color='black', weight='bold', zorder=5)
        else:
             # Goal position is static after reset
             pass

        # Draw Obstacles (Ground Truth using get_render_patch)
        # If number of obstacles changes or first render, recreate patches
        if len(self.obstacle_patches) != len(self.obstacles):
             # Remove any existing patches first
             for patch in self.obstacle_patches:
                 if patch is not None: patch.remove()
             self.obstacle_patches = []
             for obstacle in self.obstacles:
                 # Get patch from obstacle itself (which delegates to shape)
                 patch = obstacle.get_render_patch(alpha=0.6, zorder=3)
                 self.ax.add_patch(patch)
                 self.obstacle_patches.append(patch)
        # Update positions/angles for existing patches
        else:
             for i, obstacle in enumerate(self.obstacles):
                  # Remove the old patch and add the new one to handle updates correctly
                  # (especially for rotation in rectangles)
                  if self.obstacle_patches[i] is not None:
                       self.obstacle_patches[i].remove()
                  # Get updated patch
                  new_patch = obstacle.get_render_patch(alpha=0.6, zorder=3)
                  self.ax.add_patch(new_patch)
                  self.obstacle_patches[i] = new_patch # Store the new patch handle

        # Draw Path History
        if self.path:
            path_x, path_y = zip(*self.path)
            if self.path_line:
                 self.path_line.set_data(path_x, path_y)
            else:
                 self.path_line, = self.ax.plot(path_x, path_y, 'b-', linewidth=1.5, alpha=0.6, label='Robot Path', zorder=2)
        
        # Draw Oracle Path (if available)
        if hasattr(self, 'oracle_smoothed_path') and self.oracle_smoothed_path:
            oracle_x, oracle_y = zip(*self.oracle_smoothed_path)
            if not hasattr(self, 'oracle_path_line') or self.oracle_path_line is None:
                self.oracle_path_line, = self.ax.plot(oracle_x, oracle_y, 'g--', linewidth=2, alpha=0.5, label='Oracle Path', zorder=3)
            else:
                self.oracle_path_line.set_data(oracle_x, oracle_y)
        
        # Draw Oracle Path (if available)
        if hasattr(self, 'oracle_smoothed_path') and self.oracle_smoothed_path:
            oracle_x, oracle_y = zip(*self.oracle_smoothed_path)
            if not hasattr(self, 'oracle_path_line') or self.oracle_path_line is None:
                self.oracle_path_line, = self.ax.plot(oracle_x, oracle_y, 'g--', linewidth=2, alpha=0.5, label='Oracle Path', zorder=3)
            else:
                self.oracle_path_line.set_data(oracle_x, oracle_y)

        # --- Visualization from Controller Info (Optional - RRT/Path) ---
        if controller_info:
            # Visualize RRT Tree
            if 'rrt_nodes' in controller_info and 'rrt_parents' in controller_info and controller_info['rrt_nodes'] is not None:
                nodes = controller_info['rrt_nodes']
                parents = controller_info['rrt_parents']
                for i, p_idx in enumerate(parents):
                    if p_idx != -1 and i < len(nodes) and p_idx < len(nodes):
                        line, = self.ax.plot([nodes[i][0], nodes[p_idx][0]], [nodes[i][1], nodes[p_idx][1]],
                                            'grey', alpha=0.3, linewidth=0.5, zorder=1)
                        self.rrt_tree_lines.append(line)

            # Visualize Planned Path
            if 'planned_path' in controller_info and controller_info['planned_path']:
                 path_points = controller_info['planned_path']
                 if len(path_points) > 1:
                     path_x, path_y = zip(*path_points)
                     self.planned_path_line, = self.ax.plot(path_x, path_y, 'r--', linewidth=2, alpha=0.7, label='Planned Path', zorder=4)


        # Draw Sensor Range
        if self.sensor_circle is None:
             self.sensor_circle = patches.Circle((self.robot_x, self.robot_y), self.sensor_range, fc='none', ec='purple', ls=':', alpha=0.5, label='Sensor Range', zorder=2)
             self.ax.add_patch(self.sensor_circle)
        else:
             self.sensor_circle.center = (self.robot_x, self.robot_y)

        # Add legend if needed
        handles, labels = self.ax.get_legend_handles_labels()
        # Only add legend if labels exist and no legend is present yet
        # Filter out potentially duplicate labels before adding
        unique_labels_handles = {}
        for h, l in zip(handles, labels):
             if l not in unique_labels_handles:
                  unique_labels_handles[l] = h
        if unique_labels_handles and not self.ax.get_legend():
             self.ax.legend(unique_labels_handles.values(), unique_labels_handles.keys(), loc='upper right', fontsize='small')


        if mode == 'human':
            plt.draw()
            plt.pause(0.01)
            return None
        elif mode == 'rgb_array':
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image

    def close(self):
        if self.fig is not None:
            if plt.isinteractive():
                 plt.ioff()
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            # Reset all plot element handles
            self.robot_patch = None
            self.goal_patch = None
            self.goal_text = None
            self.obstacle_patches = []
            self.ray_lines = []
            self.rrt_tree_lines = []
            self.planned_path_line = None
            self.direction_arrow = None
            self.path_line = None
            self.sensor_circle = None

    def get_metrics(self):
        """Return the current metrics dictionary"""
        return self.metrics

    def get_path_metrics(self):
        """
        Returns the key path planning metrics as described in the theoretical formulas.
        
        Returns:
            dict: A dictionary containing the following metrics:
                - path_length: Total length of the path (Euclidean distance sum)
                - path_smoothness: Average angle between consecutive path segments
                - success: Binary variable indicating success (1) or failure (0)
                - spl: Success weighted by Path Length metric (1 is optimal)
                - sps: Success weighted by Path Smoothness metric (1 is optimal)
        """
        return {
            'path_length': self.metrics['path_length'],
            'path_smoothness': self.metrics['path_smoothness'],
            'success': self.metrics['success'],
            'spl': self.metrics['spl'],
            'sps': self.metrics['sps']
        }

    def set_oracle_metrics(self, oracle_shortest_path=None, oracle_smoothest_path=None):
        """
        Set the oracle metrics for better SPL and SPS calculation
        
        Args:
            oracle_shortest_path (float): The optimal shortest path length
            oracle_smoothest_path (float): The optimal smoothest path value
        """
        if oracle_shortest_path is not None:
            self.metrics['oracle_shortest_path'] = oracle_shortest_path
        
        if oracle_smoothest_path is not None:
            self.metrics['oracle_smoothest_path'] = oracle_smoothest_path
