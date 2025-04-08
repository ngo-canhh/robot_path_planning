import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from matplotlib.animation import FuncAnimation # Not strictly needed for logic
import gymnasium as gym
from gymnasium import spaces
import time
from enum import Enum
import copy # Needed for deep copying obstacles for the planner's map

class ObstacleType(Enum):
    STATIC = 0
    DYNAMIC = 1

class Obstacle:
    # --- (Obstacle class remains the same) ---
    def __init__(self, x, y, radius, obs_type=ObstacleType.STATIC, velocity=None, direction=None):
        self.x = x
        self.y = y
        self.radius = radius
        self.type = obs_type
        self.velocity = velocity if velocity is not None else 0
        # Ensure direction is a numpy array and normalized for dynamic obstacles
        if direction is not None:
            direction = np.array(direction, dtype=float)
            norm = np.linalg.norm(direction)
            if norm > 1e-6: # Avoid division by zero
                 self.direction = direction / norm
            else:
                 self.direction = np.array([0.0, 0.0])
        else:
            self.direction = np.array([0.0, 0.0])
        # Keep track of original position for reset if needed by planner
        self.initial_x = x
        self.initial_y = y

    def update(self, dt=1.0):
        if self.type == ObstacleType.DYNAMIC:
            self.x += self.velocity * self.direction[0] * dt
            self.y += self.velocity * self.direction[1] * dt

            # Simple boundary bouncing for dynamic obstacles
            if self.x - self.radius < 0 or self.x + self.radius > 500: # Assuming width=500
                self.x = np.clip(self.x, self.radius, 500 - self.radius) # Prevent getting stuck
                self.direction[0] *= -1
            if self.y - self.radius < 0 or self.y + self.radius > 500: # Assuming height=500
                self.y = np.clip(self.y, self.radius, 500 - self.radius) # Prevent getting stuck
                self.direction[1] *= -1


    def get_position(self):
        return np.array([self.x, self.y])

    def check_collision(self, x, y, robot_radius=0.5):
        distance = np.sqrt((self.x - x)**2 + (self.y - y)**2)
        return distance < (self.radius + robot_radius)

    def reset_to_initial(self):
         """Resets dynamic obstacle to its starting position for planning purposes"""
         self.x = self.initial_x
         self.y = self.initial_y


class IndoorRobotEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, width=500, height=500, robot_radius=10, max_steps=1000, sensor_range=150, render_mode='rgb_array'): # Added sensor_range
        super(IndoorRobotEnv, self).__init__()

        self.width = width
        self.height = height
        self.robot_radius = robot_radius
        self.max_steps = max_steps
        self.current_step = 0
        self.sensor_range = sensor_range # Robot's sensor range
        self.render_mode = render_mode

        self.robot_x = None
        self.robot_y = None
        self.robot_orientation = None
        self.robot_velocity = 0 # Store current velocity in state if needed by controller
        self.goal_x = None
        self.goal_y = None

        self.obstacles = [] # Ground truth obstacles

        # Observation space: [robot_x, robot_y, robot_orientation, goal_x, goal_y, sensed_obstacle_info...]
        # Sensed obstacle info format remains the same, but only includes *visible* obstacles.
        self.max_obstacles_in_observation = 10 # Max obstacles reported in one observation
        obs_low = np.array([0, 0, -np.pi, 0, 0] + [0, 0, 0, 0, -np.inf, -np.inf] * self.max_obstacles_in_observation, dtype=np.float32)
        obs_high = np.array([width, height, np.pi, width, height] + [width, height, np.inf, 1, np.inf, np.inf] * self.max_obstacles_in_observation, dtype=np.float32)
        # Increased velocity limits slightly in obs space just in case
        # Radius can technically be larger than 50, use inf

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action space: [velocity, steering_angle] - Limit velocity slightly
        self.action_space = spaces.Box(low=np.array([0, -np.pi/4]),
                                      high=np.array([10, np.pi/4]), # Max vel 10
                                      dtype=np.float32)

        self.fig = None
        self.ax = None
        self.robot_patch = None
        self.goal_patch = None
        self.obstacle_patches = []
        self.path = []
        self.ray_lines = [] # For visualizing rays if needed
        self.rrt_tree_lines = [] # For visualizing RRT tree if needed
        # Add members for controller visualization lines if not present
        self.planned_path_line = None
        self.direction_arrow = None
        self.path_line = None
        self.sensor_circle = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for gym compatibility
        self.current_step = 0

        margin = self.robot_radius + 10
        self.robot_x = self.np_random.uniform(margin, self.width - margin)
        self.robot_y = self.np_random.uniform(margin, self.height - margin)
        self.robot_orientation = self.np_random.uniform(-np.pi, np.pi)
        self.robot_velocity = 0

        # Set random goal position ensuring minimum distance from start
        min_start_goal_dist = 250 # Increased distance
        while True:
            self.goal_x = self.np_random.uniform(margin, self.width - margin)
            self.goal_y = self.np_random.uniform(margin, self.height - margin)
            if np.sqrt((self.goal_x - self.robot_x)**2 + (self.goal_y - self.robot_y)**2) >= min_start_goal_dist:
                 # Ensure goal is not inside an obstacle's initial position
                 goal_clear = True
                 # No obstacles yet at this stage in reset, check needs to happen after obstacle generation OR ensure obstacles don't spawn on goal
                 # Let's check later
                 break

        # Generate random obstacles (ground truth)
        self.obstacles = []
        num_obstacles = self.np_random.integers(5, self.max_obstacles_in_observation + 1) # Control number slightly

        for _ in range(num_obstacles):
            attempts = 0
            while True: # Ensure obstacle placement is valid
                attempts += 1
                if attempts > 100: # Prevent infinite loop if space is too crowded
                    print("Warning: Could not place obstacle, space might be too crowded.")
                    break # Skip this obstacle

                obs_x = self.np_random.uniform(margin, self.width - margin)
                obs_y = self.np_random.uniform(margin, self.height - margin)
                obs_radius = self.np_random.uniform(15, 40) # Slightly larger radii possible

                # Check minimum distance from robot start and goal
                dist_to_start = np.sqrt((obs_x - self.robot_x)**2 + (obs_y - self.robot_y)**2)
                dist_to_goal = np.sqrt((obs_x - self.goal_x)**2 + (obs_y - self.goal_y)**2)

                # Check minimum distance from other obstacles
                clear_of_others = True
                for existing_obs in self.obstacles:
                    dist_to_existing = np.sqrt((obs_x - existing_obs.x)**2 + (obs_y - existing_obs.y)**2)
                    # Ensure spacing between obstacle boundaries + robot radius buffer
                    if dist_to_existing < obs_radius + existing_obs.radius + self.robot_radius:
                        clear_of_others = False
                        break

                # Combine checks: ensure obstacle is clear of start, goal, and other obstacles
                # Added robot_radius buffer to start/goal checks
                if (dist_to_start > obs_radius + self.robot_radius + 10 and # 10 is extra margin
                    dist_to_goal > obs_radius + self.robot_radius + 10 and # 10 is extra margin
                    clear_of_others):
                    break # Valid placement found

            if attempts > 100: continue # Go to next obstacle if placement failed

            if self.np_random.random() < 0.3: # 30% chance dynamic
                obs_type = ObstacleType.DYNAMIC
                obs_velocity = self.np_random.uniform(1.0, 4.0) # Slightly faster range
                angle = self.np_random.uniform(0, 2*np.pi)
                obs_direction = np.array([np.cos(angle), np.sin(angle)])
            else:
                obs_type = ObstacleType.STATIC
                obs_velocity = 0
                obs_direction = np.array([0, 0])

            obstacle = Obstacle(obs_x, obs_y, obs_radius, obs_type, obs_velocity, obs_direction)
            self.obstacles.append(obstacle)

        # Now, double-check if goal ended up inside a generated obstacle (unlikely with checks, but safe)
        for obs in self.obstacles:
             if obs.check_collision(self.goal_x, self.goal_y, 1): # Check goal point collision
                  print("Warning: Goal spawned inside an obstacle, attempting to move goal.")
                  # Simple fix: move goal slightly away (could be more robust)
                  vec_from_obs = np.array([self.goal_x - obs.x, self.goal_y - obs.y])
                  dist = np.linalg.norm(vec_from_obs)
                  if dist < 1e-6: # Goal exactly at center
                      self.goal_x += obs.radius + 5
                  else:
                      move_dist = obs.radius + 5 - dist
                      self.goal_x += vec_from_obs[0] / dist * move_dist
                      self.goal_y += vec_from_obs[1] / dist * move_dist
                  # Clamp goal to bounds
                  self.goal_x = np.clip(self.goal_x, margin, self.width - margin)
                  self.goal_y = np.clip(self.goal_y, margin, self.height - margin)
                  # No need to recheck against all obstacles in this simple fix


        self.path = [(self.robot_x, self.robot_y)]

        observation = self._get_observation()
        info = self._get_info()

        # Clear visualization elements if they exist
        if self.ax:
             for patch in self.obstacle_patches: patch.remove()
             self.obstacle_patches = []
             if self.robot_patch: self.robot_patch.remove()
             if self.goal_patch: self.goal_patch.remove()
             if self.direction_arrow: self.direction_arrow.remove()
             if self.path_line: self.path_line.set_data([], [])
             if self.planned_path_line: self.planned_path_line.set_data([], []) # Clear planned path line
             if self.sensor_circle: self.sensor_circle.remove() # Clear sensor circle
             for line in self.ray_lines: line.remove()
             self.ray_lines = []
             for line in self.rrt_tree_lines: line.remove()
             self.rrt_tree_lines = []

             # Reset handles to None so they are recreated
             self.robot_patch = None
             self.goal_patch = None
             self.direction_arrow = None
             self.path_line = None
             self.planned_path_line = None
             self.sensor_circle = None


        return observation, info

    def _get_observation(self):
        # Base observation: robot state and goal state
        base_obs = [self.robot_x, self.robot_y, self.robot_orientation, self.goal_x, self.goal_y]

        # Add *sensed* obstacle information within sensor_range
        sensed_obstacles_data = []
        count = 0
        for obstacle in self.obstacles:
            dist_to_robot = np.sqrt((obstacle.x - self.robot_x)**2 + (obstacle.y - self.robot_y)**2)

            # Check if obstacle center is within sensor range (simplification)
            # A more realistic check would see if ANY part of the obstacle is within range
            # e.g., dist_to_robot <= self.sensor_range + obstacle.radius
            if dist_to_robot <= self.sensor_range and count < self.max_obstacles_in_observation:
                dynamic_flag = 1.0 if obstacle.type == ObstacleType.DYNAMIC else 0.0
                vel_x = obstacle.velocity * obstacle.direction[0] if obstacle.type == ObstacleType.DYNAMIC else 0.0
                vel_y = obstacle.velocity * obstacle.direction[1] if obstacle.type == ObstacleType.DYNAMIC else 0.0
                sensed_obstacles_data.extend([obstacle.x, obstacle.y, obstacle.radius, dynamic_flag, vel_x, vel_y])
                count += 1

        # Pad observation if fewer than max_obstacles_in_observation are sensed
        num_missing = self.max_obstacles_in_observation - count
        if num_missing > 0:
            # Use zeros for padding, consistent with observation space definition
            sensed_obstacles_data.extend([0.0] * 6 * num_missing)

        # Combine and ensure correct type
        observation = np.array(base_obs + sensed_obstacles_data, dtype=np.float32)

        # Verify observation shape matches space definition
        if observation.shape != self.observation_space.shape:
             # This should ideally not happen with the padding logic, but good to check
             print(f"FATAL Error: Observation shape mismatch. Got {observation.shape}, expected {self.observation_space.shape}")
             # Attempt to fix if possible (e.g., truncate/pad again), but indicates a deeper issue
             # Pad or truncate to the correct shape as a last resort
             expected_len = self.observation_space.shape[0]
             current_len = len(observation)
             if current_len > expected_len:
                 observation = observation[:expected_len]
             elif current_len < expected_len:
                 observation = np.pad(observation, (0, expected_len - current_len), 'constant', constant_values=0.0)
             print(f"Attempted to fix observation shape to {observation.shape}")


        return observation

    def _get_info(self):
        # Provide additional info not part of the observation space
        # Ground truth can be useful for debugging or visualization outside the agent's knowledge
        current_dist_to_goal = np.sqrt((self.robot_x - self.goal_x)**2 + (self.robot_y - self.goal_y)**2)
        return {
            "distance_to_goal": current_dist_to_goal,
            "ground_truth_obstacles": self.obstacles # For visualization/debug
        }


    def step(self, action):
        self.current_step += 1

        # Store distance before move for reward calculation
        prev_distance = np.sqrt((self.robot_x - self.goal_x)**2 + (self.robot_y - self.goal_y)**2)


        # --- Action Application and Collision Checking (mostly same) ---
        velocity, steering_angle = action
        # Apply physics limits maybe? Clamp velocity based on action space?
        velocity = np.clip(velocity, self.action_space.low[0], self.action_space.high[0])
        steering_angle = np.clip(steering_angle, self.action_space.low[1], self.action_space.high[1])

        # Update orientation (simple model: steering adds to orientation directly)
        # More realistic model (like Bicycle model) would depend on velocity and wheelbase
        dt = 0.5 # Assume dt=1 for simulation step time
        self.robot_orientation += steering_angle * dt # Scale change by dt if not 1
        # Normalize orientation to [-pi, pi]
        self.robot_orientation = np.arctan2(np.sin(self.robot_orientation), np.cos(self.robot_orientation))

        # Update position
        dx = velocity * np.cos(self.robot_orientation) * dt
        dy = velocity * np.sin(self.robot_orientation) * dt
        new_x = self.robot_x + dx
        new_y = self.robot_y + dy
        self.robot_velocity = velocity # Store current velocity

        terminated = False # Gym standard: episode ends due to goal, crash, etc.
        truncated = False # Gym standard: episode ends due to time limit
        reward = 0 # Initialize reward
        info = {'status': 'in_progress'} # Default status


        # Check boundary collision
        if not (self.robot_radius <= new_x <= self.width - self.robot_radius and
                self.robot_radius <= new_y <= self.height - self.robot_radius):
            reward = -50 # More severe penalty
            terminated = True
            info['status'] = 'boundary_collision'
            # Don't update position if collision
            observation = self._get_observation() # Get obs at current (pre-collision) state
            info.update(self._get_info()) # Add standard info
            return observation, reward, terminated, truncated, info

        # Check obstacle collision (using ground truth obstacles for physics)
        collision = False
        for obstacle in self.obstacles:
            if obstacle.check_collision(new_x, new_y, self.robot_radius):
                collision = True
                break

        if collision:
            reward = -50 # More severe penalty
            terminated = True
            info['status'] = 'obstacle_collision'
             # Don't update position if collision
            observation = self._get_observation() # Get obs at current (pre-collision) state
            info.update(self._get_info())
            return observation, reward, terminated, truncated, info

        # --- Update State if No Collision ---
        self.robot_x = new_x
        self.robot_y = new_y
        self.path.append((self.robot_x, self.robot_y))

        # Update dynamic obstacles (ground truth)
        for obstacle in self.obstacles:
             obstacle.update(dt=dt) # Use same dt


        # --- Calculate Reward and Termination/Truncation ---
        distance_to_goal = np.sqrt((self.robot_x - self.goal_x)**2 + (self.robot_y - self.goal_y)**2)

        # Check if goal reached
        goal_threshold = self.robot_radius + 5 # Reach within robot radius + small margin
        if distance_to_goal < goal_threshold:
            reward = 200 # Large reward for reaching goal
            terminated = True
            info['status'] = 'goal_reached'
        else:
            # Reward based on progress towards goal
            reward_dist = prev_distance - distance_to_goal # Positive if moved closer
            # Penalize distance, reward closeness (Alternative/Combined)
            # reward_dist_penalty = -0.1 * distance_to_goal

            # Time penalty
            reward_time = -0.5 # Increased time penalty

            # Control effort penalty (optional)
            # reward_action = -0.01 * (velocity**2 + steering_angle**2)

            reward = (reward_dist * 1.5) + reward_time # Emphasize progress slightly more
            #reward = reward_dist_penalty + reward_time # Alternative: Penalize distance

            # Check for truncation (max steps)
            if self.current_step >= self.max_steps:
                truncated = True
                reward -= 50 # Penalty for running out of time
                info['status'] = 'max_steps_reached'


        observation = self._get_observation() # Get obs AFTER state update
        info.update(self._get_info()) # Add standard info like distance

        return observation, reward, terminated, truncated, info

    # --- Rendering (Modified to handle optional elements) ---
    def render(self, mode='human', controller_info=None):
        if mode not in self.metadata['render.modes']:
             raise ValueError(f"Unsupported render mode: {mode}")

        if self.fig is None and mode == 'human':
            plt.ion() # Turn interactive mode on for human rendering
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.ax.set_aspect('equal')
            self.ax.grid(True)
            plt.title("Indoor Robot Simulation")
            plt.xlabel("X")
            plt.ylabel("Y")
        elif self.fig is None and mode == 'rgb_array':
            # For rgb_array, we might need a non-interactive backend
            import matplotlib
            matplotlib.use('Agg') # Use non-interactive backend
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.ax.set_aspect('equal')
            self.ax.grid(True)
            plt.title("Indoor Robot Simulation")
            plt.xlabel("X")
            plt.ylabel("Y")


        # Clear previous dynamic elements (like rays, RRT tree, planned path)
        for line in self.ray_lines: line.remove()
        self.ray_lines = []
        for line in self.rrt_tree_lines: line.remove()
        self.rrt_tree_lines = []
        if self.planned_path_line:
             self.planned_path_line.remove()
             self.planned_path_line = None


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
        if self.direction_arrow:
            self.direction_arrow.remove() # Remove old arrow
        self.direction_arrow = self.ax.arrow(self.robot_x, self.robot_y,
                                            end_x - self.robot_x, end_y - self.robot_y,
                                            head_width=max(self.robot_radius * 0.4, 3),
                                            head_length=max(self.robot_radius * 0.6, 5),
                                            fc='red', ec='red', length_includes_head=True, zorder=6)

        # Draw Goal
        if self.goal_patch is None:
            self.goal_patch = patches.Circle((self.goal_x, self.goal_y), self.robot_radius * 0.8, fc='lime', alpha=0.8, ec='green', lw=2, zorder=4)
            self.ax.add_patch(self.goal_patch)
            self.goal_text = self.ax.text(self.goal_x, self.goal_y, 'G', ha='center', va='center', color='black', weight='bold', zorder=5)
        else:
             # Goal position should not change during an episode after reset
             # self.goal_patch.center = (self.goal_x, self.goal_y) # Not needed usually
             # self.goal_text.set_position((self.goal_x, self.goal_y)) # Not needed usually
             pass # Keep goal fixed


        # Draw Obstacles (Ground Truth for visualization)
        # Recreate patches if number changes or first time
        if len(self.obstacle_patches) != len(self.obstacles):
             for patch in self.obstacle_patches: patch.remove()
             self.obstacle_patches = []
             for obstacle in self.obstacles:
                  color = 'dimgray' if obstacle.type == ObstacleType.STATIC else 'darkorange'
                  patch = patches.Circle((obstacle.x, obstacle.y), obstacle.radius, fc=color, alpha=0.6, zorder=3)
                  self.ax.add_patch(patch)
                  self.obstacle_patches.append(patch)
        # Update positions for existing patches
        else:
             for i, obstacle in enumerate(self.obstacles):
                  self.obstacle_patches[i].center = (obstacle.x, obstacle.y)


        # Draw Path History
        if self.path:
            path_x, path_y = zip(*self.path)
            if self.path_line:
                 self.path_line.set_data(path_x, path_y)
            else:
                 # Create path line only if not created before
                 self.path_line, = self.ax.plot(path_x, path_y, 'b-', linewidth=1.5, alpha=0.6, label='Robot Path', zorder=2)
                 # # Add legend only once
                 # handles, labels = self.ax.get_legend_handles_labels()
                 # if "Robot Path" not in labels: # Check if label already exists
                 #      self.ax.legend(loc='upper right', fontsize='small')

        # --- Visualization from Controller Info (Optional) ---
        if controller_info:
            # Visualize Rays (Currently not generated by default controller, but hook exists)
            if 'rays' in controller_info and controller_info['rays']:
                 for start, end in controller_info['rays']:
                      line, = self.ax.plot([start[0], end[0]], [start[1], end[1]], 'pink', alpha=0.7, linewidth=0.5, zorder=1)
                      self.ray_lines.append(line)

            # Visualize RRT Tree (from planning phase)
            if 'rrt_nodes' in controller_info and 'rrt_parents' in controller_info:
                nodes = controller_info['rrt_nodes']
                parents = controller_info['rrt_parents']
                for i, p_idx in enumerate(parents):
                    if p_idx != -1 and i < len(nodes) and p_idx < len(nodes): # Check indices are valid
                        line, = self.ax.plot([nodes[i][0], nodes[p_idx][0]], [nodes[i][1], nodes[p_idx][1]],
                                            'grey', alpha=0.3, linewidth=0.5, zorder=1)
                        self.rrt_tree_lines.append(line)

            # Visualize Planned Path (from RRT)
            if 'planned_path' in controller_info and controller_info['planned_path']:
                 path_points = controller_info['planned_path']
                 if len(path_points) > 1: # Need at least 2 points to draw a line
                     path_x, path_y = zip(*path_points)
                     # Create or update the planned_path_line
                     # Important: Create a *new* line object each time the path changes significantly
                     # because the number of points might change. Re-using with set_data is tricky here.
                     # So, we remove the old one (done above) and create a new one.
                     self.planned_path_line, = self.ax.plot(path_x, path_y, 'g--', linewidth=2, alpha=0.7, label='Planned Path', zorder=4)
                     # handles, labels = self.ax.get_legend_handles_labels()
                     # if "Planned Path" not in labels:
                     #      self.ax.legend(loc='upper right', fontsize='small')


        # Draw Sensor Range (optional)
        if self.sensor_circle is None:
             self.sensor_circle = patches.Circle((self.robot_x, self.robot_y), self.sensor_range, fc='none', ec='purple', ls=':', alpha=0.5, label='Sensor Range', zorder=2)
             self.ax.add_patch(self.sensor_circle)
             # handles, labels = self.ax.get_legend_handles_labels()
             # if "Sensor Range" not in labels:
             #      self.ax.legend(loc='upper right', fontsize='small')
        else:
             self.sensor_circle.center = (self.robot_x, self.robot_y)

        # Add legend if any labels were set and legend doesn't exist yet
        handles, labels = self.ax.get_legend_handles_labels()
        if labels and not self.ax.get_legend():
             self.ax.legend(loc='upper right', fontsize='small')


        if mode == 'human':
            plt.draw()
            plt.pause(0.01) # Short pause to allow plot to update
            return None
        elif mode == 'rgb_array':
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image

    def close(self):
        if self.fig is not None:
            if plt.isinteractive():
                 plt.ioff() # Turn off interactive mode
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.robot_patch = None
            self.goal_patch = None
            self.obstacle_patches = []
            self.ray_lines = []
            self.rrt_tree_lines = []
            # Reset other plot elements too
            self.planned_path_line = None
            self.direction_arrow = None
            self.path_line = None
            self.sensor_circle = None


class ObstacleIdentifier:
    def __init__(self, env_observation_space, max_obstacles_in_observation):
        self.observation_space = env_observation_space
        self.max_obstacles = max_obstacles_in_observation
        # In a real implementation, load GoogLeNet or other model here
        # self.detection_threshold = 0.7

    def identify(self, observation):
        """
        Identify obstacles from the limited sensor data in the observation.

        Args:
            observation: The observation numpy array from the environment.

        Returns:
            List of perceived Obstacle objects.
        """
        perceived_obstacles = []
        # Observation format: [robot_x, robot_y, robot_orientation, goal_x, goal_y, obs1_data, obs2_data, ...]
        obs_data_start_index = 5
        obstacle_data_size = 6 # x, y, radius, is_dynamic, vel_x, vel_y

        for i in range(self.max_obstacles):
            base_idx = obs_data_start_index + i * obstacle_data_size
            # Check if index is within bounds (important!)
            if base_idx + obstacle_data_size > len(observation):
                 # print(f"Warning: Index out of bounds when parsing observation for obstacle {i}")
                 break # Stop parsing if we go beyond observation length

            obs_x = observation[base_idx]
            obs_y = observation[base_idx + 1]
            obs_radius = observation[base_idx + 2]

            # Check if it's valid obstacle data (radius > 0 is a good indicator for padding)
            if obs_radius > 1e-3: # Use a small threshold instead of == 0 for float comparison
                is_dynamic = observation[base_idx + 3] > 0.5
                vel_x = observation[base_idx + 4]
                vel_y = observation[base_idx + 5]

                obstacle_type = ObstacleType.DYNAMIC if is_dynamic else ObstacleType.STATIC
                velocity = np.sqrt(vel_x**2 + vel_y**2) if is_dynamic else 0.0
                direction = np.array([vel_x, vel_y])
                if velocity > 1e-6:
                    direction = direction / velocity
                else:
                    direction = np.array([0.0, 0.0])

                # Create an Obstacle object representing the perception
                # Note: This perceived obstacle's state (x,y,vel) is instantaneous from the sensor.
                # The controller might need to track it or predict its future state separately.
                perceived_obstacles.append(Obstacle(obs_x, obs_y, obs_radius, obstacle_type, velocity, direction))

        return perceived_obstacles


class RayTracingAlgorithm:
    def __init__(self, env_width, env_height, robot_radius):
        # No direct env dependency needed if dimensions/obstacles are passed
        self.width = env_width
        self.height = env_height
        self.robot_radius = robot_radius # Needed for avoidance logic maybe
        self.num_rays = 36
        self.max_ray_length = 150 # Should maybe match sensor_range

    def trace_rays(self, robot_x, robot_y, robot_orientation, obstacles):
        """
        Implement ray tracing using the provided list of obstacles.

        Args:
            robot_x, robot_y, robot_orientation: Current robot state.
            obstacles: List of *perceived* Obstacle objects.

        Returns:
            List of tuples: (intersection_point_x, intersection_point_y, intersected_obstacle_or_None)
            Also returns ray start/end points for visualization.
        """
        ray_intersections = []
        ray_viz_points = [] # For rendering

        for i in range(self.num_rays):
            # --- Ray casting logic (mostly same, but uses passed obstacles) ---
            angle = robot_orientation + i * (2 * np.pi / self.num_rays)
            # Normalize angle to [-pi, pi] or [0, 2*pi]
            angle = np.arctan2(np.sin(angle), np.cos(angle))

            ray_dir_x = np.cos(angle)
            ray_dir_y = np.sin(angle)

            closest_intersection_dist = self.max_ray_length
            closest_intersection_point = None
            intersected_obstacle = None

            # Check intersection with perceived obstacles
            for obstacle in obstacles:
                # Vector from robot to obstacle center
                oc_x = obstacle.x - robot_x
                oc_y = obstacle.y - robot_y
                oc_dist_sq = oc_x**2 + oc_y**2

                # Projection of oc onto ray direction
                proj = oc_x * ray_dir_x + oc_y * ray_dir_y

                # Optimization: If obstacle center is behind the ray origin and outside its radius, skip
                if proj < 0 and oc_dist_sq > obstacle.radius**2:
                     continue
                # More aggressive check: if furthest point of obstacle along ray direction is behind origin
                # This check was slightly wrong before, corrected version:
                # If proj is negative, it means center is behind. If proj + radius < 0, means the whole circle is behind.
                # if proj + obstacle.radius < 0 and oc_dist_sq > obstacle.radius**2 : # Ensure origin isn't inside
                #      continue # Skip if obstacle entirely behind ray origin

                # Distance squared from obstacle center to ray line
                # Ensure projection is non-negative before this check? No, perp_dist is correct regardless.
                perp_dist_sq = oc_dist_sq - proj**2

                # Check if ray misses the obstacle circle entirely (using effective radius)
                # We should check against the obstacle's actual radius here for intersection distance
                if perp_dist_sq > obstacle.radius**2 + 1e-6: # Add epsilon for float safety
                    continue

                # Calculate distance along the ray to the intersection points
                # Handle potential floating point error giving small negative under sqrt
                half_chord_sq = obstacle.radius**2 - perp_dist_sq
                if half_chord_sq < 0: half_chord_sq = 0 # Clamp to zero

                half_chord = np.sqrt(half_chord_sq)

                # Two potential intersection distances along the ray from origin
                t1 = proj - half_chord # First intersection point (closer one if origin outside)
                # t2 = proj + half_chord # Second intersection point

                # Consider intersection only if it's in front (t1 >= 0) and closer than current closest
                if t1 >= -1e-6 and t1 < closest_intersection_dist: # Allow slightly negative t1 due to float errors if origin is near edge
                    closest_intersection_dist = t1
                    intersected_obstacle = obstacle # Store the obstacle that was hit
                # Also handle case where robot starts inside obstacle: t1<0, t2>0
                # In this case, the "first" intersection in front of the robot is at t2.
                # elif t1 < 0 and (proj + half_chord) > -1e-6 and (proj + half_chord) < closest_intersection_dist:
                     # closest_intersection_dist = proj + half_chord
                     # intersected_obstacle = obstacle


            # Check for boundary intersections
            boundary_ts = []
            # Left boundary (x=0)
            if abs(ray_dir_x) > 1e-6 and ray_dir_x < 0: # Heading towards left wall
                 t_bound = -robot_x / ray_dir_x
                 if 0 <= t_bound < self.max_ray_length: # Check within range limit too
                      y_intersect = robot_y + t_bound * ray_dir_y
                      if 0 <= y_intersect <= self.height:
                           boundary_ts.append(t_bound)
            # Right boundary (x=width)
            if abs(ray_dir_x) > 1e-6 and ray_dir_x > 0: # Heading towards right wall
                 t_bound = (self.width - robot_x) / ray_dir_x
                 if 0 <= t_bound < self.max_ray_length:
                      y_intersect = robot_y + t_bound * ray_dir_y
                      if 0 <= y_intersect <= self.height:
                           boundary_ts.append(t_bound)
             # Bottom boundary (y=0)
            if abs(ray_dir_y) > 1e-6 and ray_dir_y < 0: # Heading towards bottom wall
                 t_bound = -robot_y / ray_dir_y
                 if 0 <= t_bound < self.max_ray_length:
                      x_intersect = robot_x + t_bound * ray_dir_x
                      if 0 <= x_intersect <= self.width:
                           boundary_ts.append(t_bound)
            # Top boundary (y=height)
            if abs(ray_dir_y) > 1e-6 and ray_dir_y > 0: # Heading towards top wall
                 t_bound = (self.height - robot_y) / ray_dir_y
                 if 0 <= t_bound < self.max_ray_length:
                      x_intersect = robot_x + t_bound * ray_dir_x
                      if 0 <= x_intersect <= self.width:
                           boundary_ts.append(t_bound)

            # Find the closest boundary intersection distance
            if boundary_ts:
                 min_boundary_t = min(boundary_ts)
                 # If boundary is closer than the closest obstacle found so far
                 if min_boundary_t < closest_intersection_dist:
                      closest_intersection_dist = min_boundary_t
                      intersected_obstacle = None # Boundary hit, not an obstacle

            # Calculate the final intersection point based on the determined closest distance
            # Ensure distance is not negative due to float errors near boundary/obstacle edge
            closest_intersection_dist = max(0, closest_intersection_dist)
            closest_intersection_point = (
                robot_x + ray_dir_x * closest_intersection_dist,
                robot_y + ray_dir_y * closest_intersection_dist
            )

            ray_intersections.append((*closest_intersection_point, intersected_obstacle))
            # Store start/end points for visualization
            ray_viz_points.append(((robot_x, robot_y), closest_intersection_point))


        return ray_intersections, ray_viz_points

    def avoid_static_obstacle(self, robot_x, robot_y, robot_orientation, obstacle, goal_x, goal_y):
        """
        Suggests an avoidance orientation based on a *single* static obstacle.
        Tries to turn towards the side that is more 'open' or closer to the goal direction.

        Args:
            robot_x, robot_y, robot_orientation: Current robot state.
            obstacle: The *perceived* static Obstacle object to avoid.
            goal_x, goal_y: Target goal position.

        Returns:
            Suggested avoidance orientation (radians).
        """
        # Vector from robot to obstacle
        vec_to_obstacle = np.array([obstacle.x - robot_x, obstacle.y - robot_y])
        dist_to_obstacle_center = np.linalg.norm(vec_to_obstacle)

        if dist_to_obstacle_center < 1e-6: # Robot is on top of obstacle center
            # Turn 90 degrees from current orientation as an emergency maneuver
            return self._normalize_angle(robot_orientation + np.pi / 2)

        vec_to_obstacle_norm = vec_to_obstacle / dist_to_obstacle_center

        # Calculate angle to obstacle center
        angle_to_obstacle = np.arctan2(vec_to_obstacle_norm[1], vec_to_obstacle_norm[0])

        # Calculate two perpendicular directions (tangential to circle centered at robot, through obs center)
        angle_perp1 = self._normalize_angle(angle_to_obstacle + np.pi / 2)
        angle_perp2 = self._normalize_angle(angle_to_obstacle - np.pi / 2)

        # Vector towards goal
        vec_to_goal = np.array([goal_x - robot_x, goal_y - robot_y])
        # Avoid division by zero if goal is reached
        dist_to_goal = np.linalg.norm(vec_to_goal)
        if dist_to_goal < 1e-6:
             # If at goal, no preferred avoidance direction based on goal, pick one arbitrarily (e.g., perp1)
             return angle_perp1

        angle_to_goal = np.arctan2(vec_to_goal[1], vec_to_goal[0])

        # Choose the perpendicular direction that is angularly closer to the goal direction
        # Calculate the absolute angular difference, handling wrapping correctly
        diff1 = abs(self._normalize_angle(angle_perp1 - angle_to_goal))
        diff2 = abs(self._normalize_angle(angle_perp2 - angle_to_goal))

        # Choose the direction with the smaller angular difference to the goal
        avoidance_orientation = angle_perp1 if diff1 < diff2 else angle_perp2

        return avoidance_orientation

    def _normalize_angle(self, angle):
        """Normalize angle to be between -pi and pi"""
        while angle > np.pi: angle -= 2 * np.pi
        while angle < -np.pi: angle += 2 * np.pi
        return angle


class WaitingRule:
    def __init__(self, robot_radius, safety_margin=20, prediction_horizon=15, time_step=0.5):
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin # Extra buffer distance
        # self.max_wait_time = 5.0 # Max seconds to decide to wait in one go
        self.prediction_horizon = prediction_horizon # Number of steps to predict ahead
        self.time_step = time_step # Time duration of each prediction step (needs coordination with env step?)


    def check_dynamic_collisions(self, robot_x, robot_y, robot_velocity, robot_orientation, dynamic_obstacles):
        """
        Predicts future positions and checks for potential collisions with dynamic obstacles.

        Args:
            robot_x, robot_y: Current robot position.
            robot_velocity, robot_orientation: Intended robot velocity/direction for the next step(s).
            dynamic_obstacles: List of *perceived* dynamic Obstacle objects.

        Returns:
            List of tuples: (colliding_obstacle, time_to_collision_steps) for each predicted collision.
                           Returns empty list if no collisions predicted.
        """
        predicted_collisions = []
        # Compare squared distances for efficiency
        # Effective robot radius including safety margin
        robot_effective_radius = self.robot_radius + self.safety_margin

        for obstacle in dynamic_obstacles:
             # Ensure it's actually dynamic and moving
            if obstacle.type != ObstacleType.DYNAMIC or obstacle.velocity < 1e-3:
                 continue

            # Effective combined radius for collision check (obstacle radius + robot effective radius)
            min_separation_dist = obstacle.radius + robot_effective_radius
            min_separation_sq = min_separation_dist**2

            # Predict future states using simple linear extrapolation
            for t_step in range(1, self.prediction_horizon + 1):
                 t = t_step * self.time_step

                 # Robot predicted position based on *intended* velocity/orientation
                 future_robot_x = robot_x + robot_velocity * np.cos(robot_orientation) * t
                 future_robot_y = robot_y + robot_velocity * np.sin(robot_orientation) * t

                 # Obstacle predicted position (simple linear extrapolation)
                 # Note: This doesn't account for obstacle bouncing off walls during prediction horizon.
                 # For short horizons, this might be acceptable.
                 future_obstacle_x = obstacle.x + obstacle.velocity * obstacle.direction[0] * t
                 future_obstacle_y = obstacle.y + obstacle.velocity * obstacle.direction[1] * t

                 # Check squared distance
                 dist_sq = (future_robot_x - future_obstacle_x)**2 + (future_robot_y - future_obstacle_y)**2

                 if dist_sq < min_separation_sq:
                      predicted_collisions.append((obstacle, t_step))
                      # print(f"Predicted collision with {obstacle.x:.1f},{obstacle.y:.1f} at step {t_step}")
                      break # Found first predicted collision time with this obstacle, move to next obstacle

        return predicted_collisions


    def should_wait(self, predicted_collisions):
        """
        Simple rule: if any collision is predicted, recommend waiting.
        More complex: could check if the obstacle path clears quickly, or the collision time.
        """
        return len(predicted_collisions) > 0

    # calculate_wait_time might be less useful if we just decide to stop and reassess next step
    # def calculate_wait_time(self, robot_x, robot_y, dynamic_obstacle):
    #     """ Basic estimate: time for obstacle to move past current robot position """
    #     # ... (implementation is complex and potentially unreliable)
    #     return 1.0 # Just wait one or two steps and re-evaluate


class RRTPathPlanner:
    def __init__(self, env_width, env_height, robot_radius):
        self.width = env_width
        self.height = env_height
        self.robot_radius = robot_radius
        self.step_size = 30 # RRT step distance
        self.goal_sample_rate = 0.15 # Probability of sampling goal
        self.max_iterations = 3000 # Max RRT iterations
        self.min_dist_to_goal = self.robot_radius * 2 # Threshold to connect to goal
        # self.path_smoothing_iterations = 50 # Iterations for optional iterative smoothing (currently disabled)

        self.nodes = []
        self.parents = []
        self.planning_obstacles = [] # Obstacles used for the current plan

    def plan_path(self, start_x, start_y, goal_x, goal_y, obstacles_for_planning):
        """
        Plan path using RRT with a given set of obstacles (the robot's map).

        Args:
            start_x, start_y: Robot start position.
            goal_x, goal_y: Goal position.
            obstacles_for_planning: List of Obstacle objects known at planning time.

        Returns:
            Tuple: (path, nodes, parents)
                   path: List of (x, y) waypoints (smoothed).
                   nodes, parents: Tree nodes and parent indices for visualization.
                   Returns ([], nodes, parents) if planning fails.
        """
        # Use the provided obstacles list directly (assuming controller passes a safe copy if needed)
        self.planning_obstacles = obstacles_for_planning

        # Initialize tree
        self.nodes = [(start_x, start_y)]
        self.parents = [-1] # Parent index of root is -1
        path_found = False
        goal_node_idx = -1

        for i in range(self.max_iterations):
            # 1. Sample point
            if np.random.random() < self.goal_sample_rate:
                rnd_point = (goal_x, goal_y)
            else:
                # Sample within bounds, respecting robot radius margin
                rnd_point = (
                    np.random.uniform(self.robot_radius, self.width - self.robot_radius),
                    np.random.uniform(self.robot_radius, self.height - self.robot_radius)
                )
                # Optional: Check if random sample is inside an obstacle, resample if so
                # is_in_collision = False
                # for obs in self.planning_obstacles:
                #     if obs.check_collision(rnd_point[0], rnd_point[1], self.robot_radius):
                #         is_in_collision = True
                #         break
                # if is_in_collision: continue # Skip to next iteration if sample invalid

            # 2. Find nearest node in the tree
            nearest_idx = self._find_nearest(self.nodes, rnd_point)
            nearest_node = self.nodes[nearest_idx]

            # 3. Steer from nearest node towards sample point
            new_node = self._steer(nearest_node, rnd_point, self.step_size)

            # Ensure new node is within bounds (add margin)
            if not (self.robot_radius <= new_node[0] <= self.width - self.robot_radius and
                    self.robot_radius <= new_node[1] <= self.height - self.robot_radius):
                 continue # Skip if steered node is out of bounds

            # 4. Check collision for the new segment (using planning_obstacles)
            if self._is_collision_free(nearest_node, new_node):
                # 5. Add node and edge to tree
                self.nodes.append(new_node)
                self.parents.append(nearest_idx)
                new_node_idx = len(self.nodes) - 1

                # 6. Check if goal reached (or close enough)
                dist_to_goal = np.linalg.norm(np.array(new_node) - np.array((goal_x, goal_y)))
                if dist_to_goal <= self.min_dist_to_goal:
                     # Try connecting the new node directly to the goal
                     if self._is_collision_free(new_node, (goal_x, goal_y)):
                          # Add the actual goal node to the tree
                          self.nodes.append((goal_x, goal_y))
                          self.parents.append(new_node_idx)
                          goal_node_idx = len(self.nodes) - 1
                          print(f"RRT: Path found connecting near goal node in {i+1} iterations.")
                          path_found = True
                          break # Exit loop once goal is connected

                # Optional: Try connecting directly to goal even if not close (goal bias steer)
                # This is implicitly handled if rnd_point was the goal itself.
                # if rnd_point == (goal_x, goal_y) and dist_to_goal <= self.step_size: # If we steered towards goal and reached it
                #      goal_node_idx = new_node_idx # This node *is* the goal (or very close)
                #      print(f"RRT: Path found by steering directly to goal in {i+1} iterations.")
                #      path_found = True
                #      break


        # --- Path Extraction and Smoothing ---
        path = []
        # Keep final tree structure for visualization regardless of success
        final_nodes = list(self.nodes)
        final_parents = list(self.parents)

        if path_found:
            # Reconstruct path by backtracking from goal node
            current_idx = goal_node_idx
            while current_idx != -1:
                # Prepend node to path list
                path.insert(0, self.nodes[current_idx])
                current_idx = self.parents[current_idx]

            # print(f"RRT: Raw path length: {len(path)}")
            if len(path) > 1:
                # path = self._smooth_path(path)
                pass # Smoothing disabled for now
                # print(f"RRT: Smoothed path length: {len(path)}")
            else:
                 print("RRT: Raw path too short for smoothing.")

            # Final check if path is still valid after potential smoothing issues
            if not path:
                 print("RRT Warning: Path became empty after smoothing.")
                 # Return empty path but keep tree viz data
                 return [], final_nodes, final_parents

        else:
            print(f"RRT: Failed to find path to goal after {self.max_iterations} iterations.")
            # Return empty path but keep tree viz data
            return [], final_nodes, final_parents


        return path, final_nodes, final_parents


    def _find_nearest(self, nodes, point):
        nodes_arr = np.array(nodes)
        point_arr = np.array(point)
        distances_sq = np.sum((nodes_arr - point_arr)**2, axis=1) # Use squared distance for efficiency
        return np.argmin(distances_sq)

    def _steer(self, from_node, to_node, step_size):
        from_arr = np.array(from_node)
        to_arr = np.array(to_node)
        vec = to_arr - from_arr
        dist = np.linalg.norm(vec)

        if dist <= step_size:
            # If the target node is within step_size, return the target node itself
            return tuple(to_arr)
        else:
            # Otherwise, step towards the target node by step_size
            unit_vec = vec / dist
            new_node_arr = from_arr + unit_vec * step_size
            return tuple(new_node_arr)

    def _is_collision_free(self, from_node, to_node):
        # Check boundaries first (using robot radius margin)
        for node in [from_node, to_node]:
             x, y = node
             # Check if center is within safe area (radius away from walls)
             if not (self.robot_radius <= x <= self.width - self.robot_radius and
                      self.robot_radius <= y <= self.height - self.robot_radius):
                  # print(f"Collision check fail: Node {node} out of bounds.")
                  return False # Node itself is in collision with boundary

        # Check line segment against obstacles used during planning
        from_arr = np.array(from_node)
        to_arr = np.array(to_node)
        vec = to_arr - from_arr
        dist = np.linalg.norm(vec)

        if dist < 1e-6: return True # Nodes are essentially the same

        unit_vec = vec / dist
        # Check intermediate points along the line segment more frequently
        # Check every quarter robot radius for higher resolution check
        num_checks = max(int(dist / (self.robot_radius * 0.25)) + 1, 2)

        for i in range(num_checks + 1): # Check points including start and end (t=0 to t=1)
            t = i / num_checks
            check_point = from_arr + unit_vec * (dist * t)

            # Check point against environment boundaries (redundant if start/end check passed?)
            # No, intermediate points could go out of bounds if path clips corner
            if not (self.robot_radius <= check_point[0] <= self.width - self.robot_radius and
                      self.robot_radius <= check_point[1] <= self.height - self.robot_radius):
                 # print(f"Collision check fail: Segment point {check_point} out of bounds.")
                 return False

            # Check point against static obstacles in the planner's map
            # We assume RRT plans primarily around static map obstacles
            for obstacle in self.planning_obstacles:
                 # Use the obstacle's check_collision method with the robot's radius
                 if obstacle.check_collision(check_point[0], check_point[1], self.robot_radius):
                      # print(f"Collision check fail: {from_node} -> {to_node} at {check_point} with obs {obstacle.x, obstacle.y}")
                      return False
        return True

    def _smooth_path(self, path):
        """ Shortcut path by connecting non-adjacent waypoints if collision-free """
        if len(path) <= 2:
            return path # Cannot smooth path with 0, 1 or 2 points

        smoothed_path = [path[0]] # Start with the first node
        current_idx = 0 # Index in the ORIGINAL path

        while current_idx < len(path) - 1: # Iterate until we process the second-to-last node
            # Try connecting the node at current_idx to the furthest possible node ahead
            best_next_idx = current_idx + 1 # Default: just move to the next node in original path
            # Iterate backwards from the goal towards the node after current_idx
            for next_idx in range(len(path) - 1, current_idx + 1, -1):
                # Check if the direct connection from path[current_idx] to path[next_idx] is collision free
                if self._is_collision_free(path[current_idx], path[next_idx]):
                    # If it is free, this is the best shortcut from current_idx
                    best_next_idx = next_idx
                    break # Found the furthest connection, no need to check closer ones

            # Add the chosen next node (either a shortcut or just the next one) to smoothed path
            smoothed_path.append(path[best_next_idx])
            # Update current_idx to the index of the node just added to the smoothed path
            current_idx = best_next_idx

            # If best_next_idx was the goal (len(path) - 1), the loop condition
            # current_idx < len(path) - 1 will be false, and the loop terminates.

        # --- Removed the potentially buggy extra check for the goal here ---

        # Optional iterative smoothing pass (like B-spline idea but simpler) - currently disabled
        # Needs careful implementation and collision checking for new segments.
        # for _ in range(self.path_smoothing_iterations):
        #      # ... (implementation omitted for brevity, requires careful checking) ...
        #      pass

        return smoothed_path


class IndoorRobotController:
    def __init__(self, env):
        # Store necessary env parameters, don't keep env itself if possible
        self.width = env.width
        self.height = env.height
        self.robot_radius = env.robot_radius
        self.action_space = env.action_space # To access limits
        self.observation_space = env.observation_space # Needed for ObstacleIdentifier

        # Initialize components with necessary parameters
        self.obstacle_identifier = ObstacleIdentifier(self.observation_space, env.max_obstacles_in_observation)
        self.ray_tracer = RayTracingAlgorithm(self.width, self.height, self.robot_radius)
        self.waiting_rule = WaitingRule(self.robot_radius)
        self.path_planner = RRTPathPlanner(self.width, self.height, self.robot_radius)

        # Controller parameters
        self.goal_threshold = self.robot_radius + 5 # Increased threshold slightly
        self.max_velocity = self.action_space.high[0] # Get from action space
        self.min_velocity = 0.2 # Minimum operational velocity (avoid zero if stuck)
        self.obstacle_slow_down_distance = self.robot_radius * 5 # Start slowing earlier
        self.obstacle_avoid_distance = self.robot_radius * 3   # Distance to actively start avoiding static obs

        # Path tracking parameters
        self.lookahead_distance = self.robot_radius * 4 # Look further ahead on path
        # self.replanning_timer = 10.0 # Replan every N seconds regardless (disabled for now)
        # self.path_stale_time = 15.0 # If path is older than this, replan (disabled for now)
        self.path_invalidation_check_horizon = 5 # How many segments ahead to check for invalidation

        # Controller state
        self.current_planned_path = None
        self.current_rrt_nodes = None # For viz
        self.current_rrt_parents = None # For viz
        self.current_path_target_idx = 0 # Index of the next waypoint *target* in the current_planned_path
        self.perceived_obstacles = [] # Obstacles identified in the current step
        self.map_obstacles = [] # Obstacles used for the last RRT plan (the "map")
        self.is_waiting = False
        self.last_replanning_time = -np.inf # Force replan on first step
        self.status = "Initializing" # Add a status message

    def reset(self):
         """Resets controller state for a new episode."""
         self.current_planned_path = None
         self.current_rrt_nodes = None
         self.current_rrt_parents = None
         self.current_path_target_idx = 0
         self.perceived_obstacles = []
         self.map_obstacles = [] # Clear the map used for planning too
         self.is_waiting = False
         self.last_replanning_time = -np.inf # Ensure replan on first step of new episode
         self.status = "Reset"


    def get_action(self, observation):
        """
        Determine the next action based on the current observation and internal state.
        """
        current_time = time.time() # For potential time-based logic (currently unused)
        self.status = "Processing"

        # 1. Perceive Environment
        robot_x, robot_y, robot_orientation, goal_x, goal_y = observation[:5]
        self.perceived_obstacles = self.obstacle_identifier.identify(observation)
        # print(f"Perceived {len(self.perceived_obstacles)} obstacles.")

        # 2. Check Goal Reached
        distance_to_goal = np.linalg.norm(np.array([robot_x, robot_y]) - np.array([goal_x, goal_y]))
        if distance_to_goal < self.goal_threshold:
            self.status = "Goal Reached - Stopping"
            # print(self.status)
            # Clear path so it doesn't try to follow anymore
            self.current_planned_path = None
            self.current_path_target_idx = 0
            info = self._get_controller_info()
            info['goal_reached'] = True # Add flag for external logic if needed
            return np.array([0.0, 0.0]), info # Stop action

        # 3. Replanning Logic
        # --- Conditions for replanning ---
        needs_replan = False
        # Condition 1: No path exists or path is trivially short (only start point)
        if self.current_planned_path is None or len(self.current_planned_path) <= 1:
             self.status = "No valid path exists"
             needs_replan = True
        # Condition 2: Current target index is somehow invalid (shouldn't happen with checks)
        elif self.current_path_target_idx >= len(self.current_planned_path):
             self.status = "Target index out of bounds"
             needs_replan = True
             self.current_planned_path = None # Invalidate path if index is wrong
        # Condition 3: Check if the path ahead is blocked by newly perceived obstacles
        elif self._is_path_invalidated(robot_x, robot_y):
             self.status = "Path invalidated by new obstacle"
             needs_replan = True
        # Condition 4: Optional Timer-based replanning (currently disabled)
        # elif current_time - self.last_replanning_time > self.replanning_timer:
        #     self.status = "Periodic replan timer"
        #     needs_replan = True

        # --- Execute replanning if needed ---
        if needs_replan:
            self.status += " - Replanning..."
            print(self.status) # Keep replanning notification
            # Update the "map" for the planner. Use currently perceived static obstacles.
            # Dynamic obstacles are handled reactively, not included in the RRT plan.
            self.map_obstacles = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.STATIC]

            # Make a deep copy of the map obstacles to pass to the planner,
            # ensuring the planner doesn't modify the controller's current map instance.
            map_copy = copy.deepcopy(self.map_obstacles)

            # Call the RRT planner
            new_path, nodes, parents = self.path_planner.plan_path(robot_x, robot_y, goal_x, goal_y, map_copy)

            if new_path and len(new_path) > 1: # Check if planner returned a valid path
                 self.current_planned_path = new_path
                 self.current_rrt_nodes = nodes # Store for viz
                 self.current_rrt_parents = parents # Store for viz
                 # Reset path index to start following the new path from the beginning.
                 # The lookahead logic will find the appropriate point based on robot's current pos.
                 self.current_path_target_idx = 0
                 self.last_replanning_time = current_time
                 self.status = "Replanning Successful"
                 # print(f"New path length: {len(self.current_planned_path)}")
            else:
                 # Replanning failed
                 self.status = "Replanning Failed"
                 print(self.status)
                 # Invalidate the current path (if any) as planner couldn't find a new one from current state
                 self.current_planned_path = None
                 self.current_path_target_idx = 0
                 self.current_rrt_nodes = nodes # Store failed tree for viz if needed
                 self.current_rrt_parents = parents
                 # If replanning fails, stop the robot as we have no valid path.
                 return np.array([0.0, 0.0]), self._get_controller_info()

        # --- If still no valid path after trying to replan, stop ---
        # This check is technically redundant due to the replan failure handling above, but safe.
        if self.current_planned_path is None or len(self.current_planned_path) <= 1:
             self.status = "No valid path - Stopping"
             # print(self.status) # Avoid printing stop message every step if stuck
             return np.array([0.0, 0.0]), self._get_controller_info()


        # 4. Reactive Behavior - Dynamic Obstacles (Waiting Rule)
        # --- Check for predicted collisions with *perceived* dynamic obstacles ---
        # Predict based on the robot's *potential* max velocity to be conservative
        potential_velocity = self.max_velocity # Use max speed for collision prediction
        # Alternatively, use the velocity calculated later? Might be less safe.

        perceived_dynamic_obstacles = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.DYNAMIC]
        predicted_collisions = self.waiting_rule.check_dynamic_collisions(
            robot_x, robot_y, potential_velocity, robot_orientation, perceived_dynamic_obstacles
        )

        if self.waiting_rule.should_wait(predicted_collisions):
             colliding_obs_info = [(f"Obs(x={obs.x:.1f},y={obs.y:.1f})", t) for obs, t in predicted_collisions]
             self.status = f"Waiting for dynamic obstacle(s): {colliding_obs_info}"
             # print(self.status) # Can be noisy, print optionally
             self.is_waiting = True
             # Stop action
             return np.array([0.0, 0.0]), self._get_controller_info()
        else:
             # No predicted dynamic collision, proceed with path following/static avoidance
             self.is_waiting = False


        # 5. Path Following & Static Obstacle Avoidance
        self.status = "Following Path"

        # --- Find lookahead point on the current path ---
        lookahead_point, self.current_path_target_idx = self._get_lookahead_point(robot_x, robot_y)

        if lookahead_point is None: # Should not happen if path is valid and long enough
             self.status = "Path Error (Lookahead) - Stopping"
             print(self.status)
             self.current_planned_path = None # Force replan next step
             return np.array([0.0, 0.0]), self._get_controller_info()

        target_x, target_y = lookahead_point

        # --- Calculate target orientation towards lookahead point ---
        target_vector = np.array([target_x - robot_x, target_y - robot_y])
        target_distance = np.linalg.norm(target_vector) # Distance to the lookahead point
        # Avoid division by zero if lookahead point is exactly robot position (unlikely)
        if target_distance < 1e-6:
             # If very close to lookahead, aim for the *next* point on path if possible
             if self.current_path_target_idx < len(self.current_planned_path) - 1:
                  next_target = self.current_planned_path[self.current_path_target_idx + 1]
                  target_vector = np.array([next_target[0] - robot_x, next_target[1] - robot_y])
                  target_orientation = np.arctan2(target_vector[1], target_vector[0])
             else:
                  # At the end of the path, just maintain current orientation or aim towards goal
                  target_orientation = robot_orientation # Maintain heading
        else:
             target_orientation = np.arctan2(target_vector[1], target_vector[0])

        # --- Reactive Behavior - Static Obstacles (Proximity Check & Avoidance) ---
        # Check proximity to *perceived* static obstacles for local avoidance override
        final_target_orientation = target_orientation
        min_dist_to_static = float('inf')
        closest_static_obs_eff_dist = float('inf') # Effective distance (center_dist - radii)
        avoiding_obstacle = None

        perceived_static_obstacles = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.STATIC]
        for obstacle in perceived_static_obstacles:
             vec_to_obs = np.array([obstacle.x - robot_x, obstacle.y - robot_y])
             dist_center = np.linalg.norm(vec_to_obs)
             eff_dist = dist_center - obstacle.radius - self.robot_radius # Distance between boundaries

             # Keep track of closest obstacle overall for velocity scaling
             closest_static_obs_eff_dist = min(closest_static_obs_eff_dist, eff_dist)

             # If obstacle is close enough to potentially require avoidance maneuver
             if eff_dist < self.obstacle_avoid_distance:
                  # Check if obstacle is roughly in the direction of the *intended path target*
                  angle_to_obs = np.arctan2(vec_to_obs[1], vec_to_obs[0])
                  # Angle difference between direction to obstacle and direction to path target
                  angle_diff_to_target = abs(self._normalize_angle(angle_to_obs - target_orientation))

                  # Only trigger avoidance if obstacle is somewhat in front (e.g., within +/- 90 degrees of target direction)
                  if angle_diff_to_target < np.pi / 1.9: # Slightly wider than 90deg cone
                      # Use the closest obstacle within the avoidance cone
                      if avoiding_obstacle is None or eff_dist < min_dist_to_static:
                           min_dist_to_static = eff_dist
                           avoiding_obstacle = obstacle

        # If an obstacle requires avoidance
        if avoiding_obstacle is not None:
            self.status = f"Avoiding static obstacle near {avoiding_obstacle.x:.1f},{avoiding_obstacle.y:.1f}"
            # Calculate avoidance direction using a helper function (e.g., tangential steering)
            avoidance_orientation = self.ray_tracer.avoid_static_obstacle(
                robot_x, robot_y, robot_orientation, avoiding_obstacle, goal_x, goal_y # Pass goal as fallback target
            )

            # Blend the target orientation (from path following) with the avoidance orientation
            # Blend factor: 1 when very close to obstacle, 0 when at the edge of avoidance distance
            blend_factor = 1.0 - np.clip(min_dist_to_static / self.obstacle_avoid_distance, 0.0, 1.0)
            blend_factor = blend_factor**2 # Make blending stronger when closer

            # Use spherical linear interpolation (SLERP) for smooth angle blending
            final_target_orientation = self._slerp_angle(target_orientation, avoidance_orientation, blend_factor)
            # print(f"Blending: target={np.degrees(target_orientation):.1f}, avoid={np.degrees(avoidance_orientation):.1f}, blend={blend_factor:.2f}, final={np.degrees(final_target_orientation):.1f}")


        # 6. Calculate Control Action (Velocity and Steering)
        # --- Steering Control ---
        # Calculate the required change in orientation (steering command)
        steering_angle = self._normalize_angle(final_target_orientation - robot_orientation)
        # Clamp steering angle based on action space limits (robot's physical capability)
        steering_angle = np.clip(steering_angle, self.action_space.low[1], self.action_space.high[1])

        # --- Velocity Control ---
        # Base velocity: Use maximum allowed velocity
        velocity = self.max_velocity

        # Factor 1: Reduce velocity based on steering angle (sharper turn = slower)
        # Use cosine scaling: 1 for straight, closer to 0 for max turn
        max_steer = abs(self.action_space.high[1]) # Max steering angle magnitude
        # Normalize steering magnitude to [0, 1] relative to max steer
        norm_steer_mag = abs(steering_angle) / max_steer if max_steer > 1e-6 else 0
        # Steering velocity factor (e.g., cos(normalized_steer * pi/2))
        # Ensures factor is 1 at zero steer, 0 at max steer
        steering_vel_factor = np.cos(norm_steer_mag * np.pi / 2)
        steering_vel_factor = max(0.1, steering_vel_factor) # Ensure minimum speed even when turning max

        # Factor 2: Reduce velocity based on proximity to the *closest perceived obstacle* (static or dynamic)
        min_dist_overall_eff = closest_static_obs_eff_dist # Start with closest static
        for obs in perceived_dynamic_obstacles: # Also consider dynamic ones for slowdown
             dist_center = np.linalg.norm(np.array([robot_x, robot_y]) - np.array([obs.x, obs.y]))
             eff_dist = dist_center - obs.radius - self.robot_radius
             min_dist_overall_eff = min(min_dist_overall_eff, eff_dist)

        # Proximity velocity factor: 1 far away, 0 when touching obstacle boundary
        # Scales linearly from 0 at eff_dist=0 to 1 at eff_dist=obstacle_slow_down_distance
        proximity_vel_factor = np.clip(min_dist_overall_eff / self.obstacle_slow_down_distance, 0.0, 1.0)
        # Optional: Make slowdown curve non-linear (e.g., sqrt for gentler start)
        # proximity_vel_factor = proximity_vel_factor**0.5

        # Combine factors to determine final velocity
        velocity = self.max_velocity * steering_vel_factor * proximity_vel_factor
        # Ensure velocity is within allowed range [min_velocity, max_velocity]
        velocity = np.clip(velocity, self.min_velocity, self.max_velocity)

        # Final action
        action = np.array([velocity, steering_angle], dtype=np.float32)

        # --- Get diagnostic info ---
        controller_info = self._get_controller_info()

        # Debug print (optional)
        # print(f"Status: {self.status}, Action: Vel={action[0]:.2f}, Steer={np.degrees(action[1]):.1f}, TargetIdx: {self.current_path_target_idx}/{len(self.current_planned_path)}, Lookahead: ({lookahead_point[0]:.1f},{lookahead_point[1]:.1f})")

        return action, controller_info


    def _get_lookahead_point(self, robot_x, robot_y):
        """
        Finds a point on the planned path ahead of the robot and updates the target index.

        Returns:
            tuple: (lookahead_point, updated_target_idx) or (None, current_target_idx) if error.
        """
        if not self.current_planned_path or len(self.current_planned_path) < 2:
            return None, self.current_path_target_idx # Path is invalid or too short

        robot_pos = np.array([robot_x, robot_y])
        current_target_idx = self.current_path_target_idx # Use local copy

        # --- Update Path Target Index ---
        # Find the segment on the path the robot is currently closest to.
        # Start searching from slightly behind the current target index to handle deviations.
        search_start_idx = max(0, current_target_idx - 1)
        min_dist_to_segment_sq = float('inf')
        closest_segment_idx = current_target_idx # Default to current target segment
        projection_t = 0.0 # Projection parameter (0=start of segment, 1=end)

        for i in range(search_start_idx, len(self.current_planned_path) - 1):
             p1 = np.array(self.current_planned_path[i])
             p2 = np.array(self.current_planned_path[i+1])
             seg_vec = p2 - p1
             seg_len_sq = np.dot(seg_vec, seg_vec)

             if seg_len_sq < 1e-9: # Segment is effectively a point
                  # Distance is just distance to the point p1 (or p2)
                  dist_sq = np.dot(robot_pos - p1, robot_pos - p1)
                  t = 0.0 # Define t=0 for point segment
             else:
                  # Project robot position onto the line defined by the segment
                  # t = dot(robot - p1, p2 - p1) / |p2 - p1|^2
                  t = np.dot(robot_pos - p1, seg_vec) / seg_len_sq
                  # Clamp projection parameter to [0, 1] to find closest point *on the segment*
                  t_clamped = np.clip(t, 0, 1)
                  closest_point_on_segment = p1 + t_clamped * seg_vec
                  dist_sq = np.dot(robot_pos - closest_point_on_segment, robot_pos - closest_point_on_segment)

             # If this segment is closer than the minimum found so far
             if dist_sq < min_dist_to_segment_sq:
                  min_dist_to_segment_sq = dist_sq
                  closest_segment_idx = i
                  projection_t = t # Store the *unclamped* projection t for index update logic

        # Advance the target index if the robot has moved significantly along or past the segment
        # leading to the current target waypoint.
        # If closest segment is the one *before* the current target OR the robot projection
        # is significantly past the *start* of the segment leading to the target.
        # We use the *unclamped* t here: if t > 1, robot is past point p2 of segment i.
        # Advance if projection is past the end of the segment idx = current_target_idx - 1
        if closest_segment_idx == current_target_idx - 1 and projection_t > 1.0:
             current_target_idx = min(current_target_idx + 1, len(self.current_planned_path) - 1)
        # Advance if projection is near/past the end of the segment idx = current_target_idx
        # This needs careful thought. Let's stick to advancing based on closest segment index relative to current target.
        # If the closest segment found is AT or BEYOND the current target index, advance the target.
        # This is simpler and handles cases where the robot deviates and projection jumps.
        if closest_segment_idx >= current_target_idx:
             # Move target to the *end* waypoint of the closest segment found
             current_target_idx = min(closest_segment_idx + 1, len(self.current_planned_path) - 1)


        # Ensure target index is valid (redundant check, but safe)
        current_target_idx = min(current_target_idx, len(self.current_planned_path) - 1)

        # --- Find Lookahead Point ---
        # Starting from the *updated* target index, find a point 'lookahead_distance' away along the path.
        lookahead_point_found = None
        dist_along_path = 0.0
        # Start calculation from the waypoint *before* the current target index,
        # as the relevant segment is between (target_idx-1) and target_idx.
        start_node_idx = max(0, current_target_idx - 1)
        start_node_pos = np.array(self.current_planned_path[start_node_idx])

        # Calculate distance from robot to the start node of the segment containing the lookahead search start
        dist_to_start_node = np.linalg.norm(robot_pos - start_node_pos)


        for i in range(start_node_idx, len(self.current_planned_path) - 1):
            p1 = np.array(self.current_planned_path[i])
            p2 = np.array(self.current_planned_path[i+1])
            segment_vec = p2 - p1
            segment_len = np.linalg.norm(segment_vec)

            # Project robot onto current segment i to find how far along it we start
            if i == start_node_idx: # Only for the first segment in this loop
                 seg_len_sq = np.dot(segment_vec, segment_vec)
                 if seg_len_sq > 1e-9:
                      t = np.dot(robot_pos - p1, segment_vec) / seg_len_sq
                      dist_along_first_segment = np.clip(t, 0, 1) * segment_len
                 else:
                      dist_along_first_segment = 0
                 effective_start_dist = dist_along_first_segment # Distance covered on the starting segment
            else:
                 effective_start_dist = 0 # For subsequent segments, start count from 0


            # Check if the lookahead distance falls within the *remaining* part of this segment
            if dist_along_path + (segment_len - effective_start_dist) >= self.lookahead_distance:
                # Yes, the lookahead point is on this segment
                remaining_dist_needed = self.lookahead_distance - dist_along_path
                # Ratio needed along the current segment (from effective start)
                ratio = (effective_start_dist + remaining_dist_needed) / segment_len if segment_len > 1e-6 else 0
                ratio = np.clip(ratio, 0, 1) # Ensure ratio stays within [0,1]

                lookahead_point_found = tuple(p1 + ratio * segment_vec)
                break # Found the lookahead point

            else:
                # No, the lookahead point is further down. Add the remaining length of this segment.
                dist_along_path += (segment_len - effective_start_dist)
                # Continue to the next segment

        # If lookahead point wasn't found (e.g., path is shorter than lookahead distance)
        if lookahead_point_found is None:
            # Return the last point of the path as the lookahead point
            lookahead_point_found = self.current_planned_path[-1]

        return lookahead_point_found, current_target_idx


    def _is_path_invalidated(self, robot_x, robot_y):
        """ Checks if newly perceived obstacles block the current path significantly ahead of the robot. """
        # If no path, or path too short, or already near the end, no need to invalidate
        if not self.current_planned_path or len(self.current_planned_path) < 2 or self.current_path_target_idx >= len(self.current_planned_path) - 1:
             return False

        robot_pos = np.array([robot_x, robot_y])

        # --- Find robot's approximate position along the path ---
        # Find the closest segment index to the robot (similar logic to lookahead)
        min_dist_sq = float('inf')
        closest_segment_idx = self.current_path_target_idx # Start search near current target
        search_start_idx = max(0, self.current_path_target_idx - 1)

        for i in range(search_start_idx, len(self.current_planned_path) - 1):
             p1 = np.array(self.current_planned_path[i])
             p2 = np.array(self.current_planned_path[i+1])
             seg_vec = p2 - p1
             seg_len_sq = np.dot(seg_vec, seg_vec)
             if seg_len_sq < 1e-9:
                  dist_sq = np.dot(robot_pos - p1, robot_pos - p1)
             else:
                  t = np.clip(np.dot(robot_pos - p1, seg_vec) / seg_len_sq, 0, 1)
                  closest_point_on_segment = p1 + t * seg_vec
                  dist_sq = np.dot(robot_pos - closest_point_on_segment, robot_pos - closest_point_on_segment)

             if dist_sq < min_dist_sq:
                  min_dist_sq = dist_sq
                  closest_segment_idx = i

        # --- Check segments ahead for collisions ---
        # Check segments starting from the segment the robot is currently closest to,
        # up to a certain horizon.
        start_check_idx = closest_segment_idx
        # Check N segments ahead, or until the end of the path
        end_check_idx = min(len(self.current_planned_path) - 2, start_check_idx + self.path_invalidation_check_horizon)

        for i in range(start_check_idx, end_check_idx + 1):
             p1 = self.current_planned_path[i]
             p2 = self.current_planned_path[i+1]
             # Check this path segment against *currently perceived obstacles*
             # Use the same collision checking utility as the planner
             if not self._check_segment_collision_free(p1, p2, self.perceived_obstacles):
                 print(f"Path segment {i} ({p1[0]:.1f},{p1[1]:.1f} -> {p2[0]:.1f},{p2[1]:.1f}) blocked by perceived obstacle.")
                 return True # Path is blocked ahead

        return False # Path ahead seems clear based on current perception and horizon

    def _check_segment_collision_free(self, from_node, to_node, obstacles_to_check):
        """
        Utility to check if a line segment is collision-free against boundaries and a given list of obstacles.
        Uses the same logic as RRT's internal check.
        """
        # Check boundaries first (using robot radius margin)
        for node in [from_node, to_node]:
             x, y = node
             if not (self.robot_radius <= x <= self.width - self.robot_radius and
                      self.robot_radius <= y <= self.height - self.robot_radius):
                  return False # Node itself is in collision with boundary

        from_arr = np.array(from_node)
        to_arr = np.array(to_node)
        vec = to_arr - from_arr
        dist = np.linalg.norm(vec)

        if dist < 1e-6: return True # Nodes are the same

        unit_vec = vec / dist
        # Check intermediate points frequently
        num_checks = max(int(dist / (self.robot_radius * 0.25)) + 1, 2)

        for i in range(num_checks + 1): # Check points including start and end
            t = i / num_checks
            check_point = from_arr + unit_vec * (dist * t)

            # Check point against environment boundaries
            if not (self.robot_radius <= check_point[0] <= self.width - self.robot_radius and
                      self.robot_radius <= check_point[1] <= self.height - self.robot_radius):
                 return False

            # Check point against the provided list of obstacles
            for obstacle in obstacles_to_check:
                 if obstacle.check_collision(check_point[0], check_point[1], self.robot_radius):
                      return False # Collision detected
        return True # Segment is collision-free


    def _normalize_angle(self, angle):
        """Normalize angle to be between -pi and pi"""
        while angle > np.pi: angle -= 2 * np.pi
        while angle < -np.pi: angle += 2 * np.pi
        return angle

    def _slerp_angle(self, a1, a2, t):
        """ Spherical linear interpolation for angles (normalized to [-pi, pi])"""
        a1 = self._normalize_angle(a1)
        a2 = self._normalize_angle(a2)
        diff = self._normalize_angle(a2 - a1)
        # Interpolate the difference and add back to a1
        interpolated_angle = a1 + diff * t
        return self._normalize_angle(interpolated_angle)

    def _get_controller_info(self):
        """ Returns diagnostic info for rendering or debugging """
        info = {
            "status": self.status,
            "planned_path": self.current_planned_path, # Send current path for viz
            "rrt_nodes": self.current_rrt_nodes,       # Send tree nodes
            "rrt_parents": self.current_rrt_parents,   # Send tree structure
            "target_idx": self.current_path_target_idx # Send current target index
            }
        # Add ray tracing results if they were generated and stored
        # if hasattr(self, 'last_ray_viz_points'):
        #     info["rays"] = self.last_ray_viz_points
        return info


# --- Example Usage ---
if __name__ == "__main__":
    # Create environment
    # Use render_mode='human' for visual output, 'rgb_array' for frame grabbing
    env = IndoorRobotEnv(width=500, height=500, sensor_range=150, render_mode='human')
    # env = IndoorRobotEnv(width=500, height=500, sensor_range=150, render_mode='rgb_array') # For no display

    # Create controller
    controller = IndoorRobotController(env) # Pass env to init controller

    # Run simulation loop
    max_episodes = 5
    episode_rewards = []
    episode_steps = []
    episode_status = []

    for episode in range(max_episodes):
        print(f"\n--- Starting Episode {episode+1} ---")
        observation, info = env.reset(seed=episode) # Use different seed per episode for variety
        controller.reset() # Reset controller state for new episode
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        controller_info = None # Initialize controller info

        # Render initial state
        if env.render_mode == 'human':
            env.render(controller_info=controller_info)
            # Optional pause to see start state
            # plt.pause(1.0)


        while not terminated and not truncated:
            step_count += 1
            # Get action from controller based on current observation
            action, controller_info = controller.get_action(observation)

            # Step the environment with the chosen action
            observation, reward, terminated, truncated, info = env.step(action)

            # Render the current state (passing controller info for visualization)
            if env.render_mode == 'human':
                env.render(controller_info=controller_info)
            elif env.render_mode == 'rgb_array':
                 img = env.render(controller_info=controller_info)
                 # Process img (e.g., save to video)

            total_reward += reward

            # Optional: break early if controller status indicates a persistent failure?
            # if "Failed" in controller.status or "Error" in controller.status:
            #     print(f"Controller indicates failure state: {controller.status}, ending episode early.")
            #     break # Or set truncated=True

        # Episode finished
        status = info.get('status', 'unknown')
        print(f"Episode {episode+1} finished after {step_count} steps.")
        print(f"Status: {status}")
        print(f"Total reward: {total_reward:.2f}")
        episode_rewards.append(total_reward)
        episode_steps.append(step_count)
        episode_status.append(status)

        # Add a pause at the end of the episode if rendering human
        if env.render_mode == 'human' and (terminated or truncated):
             print("Episode end. Pausing...")
             try:
                 plt.pause(2.0) # Pause for 2 seconds
             except Exception as e:
                 print(f"Error during pause: {e}") # Handle cases where plot might be closed


    env.close()
    print("\n--- Simulation Summary ---")
    print(f"Ran {max_episodes} episodes.")
    print(f"Final Statuses: {episode_status}")
    print(f"Episode Steps: {episode_steps}")
    print(f"Episode Rewards: {[f'{r:.2f}' for r in episode_rewards]}")
    if episode_rewards:
         print(f"Average Steps: {np.mean(episode_steps):.1f}")
         print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print("--------------------------")
