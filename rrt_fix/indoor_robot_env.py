import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import AnnotationBbox
import gymnasium as gym
from gymnasium import spaces
import random
import os

from components.shape import Circle, Rectangle
from components.obstacle import StaticObstacle, DynamicObstacle, ObstacleType, find_image_files
from utils.obstacle_distinguish import GoogleNet
from utils.ray_tracing_algorithm import RayTracingAlgorithm
from utils.waiting_rule import WaitingRule

IMAGE_DIR = r"D:\robot\robot_path_planning\rrt_fix\assets\images"

print(f"Image directory set to: {IMAGE_DIR}")

class IndoorRobotEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    OBS_ROBOT_STATE_SIZE = 5
    OBS_OBSTACLE_DATA_SIZE = 9

    def __init__(self, width=500, height=500, robot_radius=10, max_steps=1000, sensor_range=250, render_mode='rgb_array'):
        super(IndoorRobotEnv, self).__init__()

        self.width = width
        self.height = height
        self.robot_radius = robot_radius
        self.max_steps = max_steps
        self.current_step = 0
        self.sensor_range = sensor_range
        self.render_mode = render_mode

        self.robot_x = None
        self.robot_y = None
        self.robot_orientation = None
        self.robot_velocity = 0
        self.goal_x = None
        self.goal_y = None

        self.obstacles = []

        # Initialize RayTracingAlgorithm and WaitingRule
        self.ray_tracer = RayTracingAlgorithm(env_width=width, env_height=height, robot_radius=robot_radius)
        self.waiting_rule = WaitingRule(robot_radius=robot_radius, safety_margin=20, prediction_horizon=15, time_step=0.5)

        try:
            self.classifier = GoogleNet()
            print("GoogleNet Classifier initialized.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to initialize GoogleNet classifier: {e}")
            self.classifier = None

        try:
            self.available_image_paths = find_image_files(IMAGE_DIR)
            if not self.available_image_paths:
                print(f"CRITICAL WARNING: No image files found in {IMAGE_DIR}.")
            else:
                print(f"Found {len(self.available_image_paths)} images for obstacles.")

                
        except Exception as e:
            print(f"CRITICAL ERROR finding image files: {e}")
            self.available_image_paths = []

        self.max_obstacles_in_observation = 10
        obs_len = self.OBS_ROBOT_STATE_SIZE + self.max_obstacles_in_observation * self.OBS_OBSTACLE_DATA_SIZE
        obs_low = np.full(obs_len, -np.inf, dtype=np.float32)
        obs_high = np.full(obs_len, np.inf, dtype=np.float32)
        obs_low[0:2] = 0.0
        obs_high[0:2] = [width, height]
        obs_low[2] = -np.pi
        obs_high[2] = np.pi
        obs_low[3:5] = 0.0
        obs_high[3:5] = [width, height]
        for i in range(self.max_obstacles_in_observation):
            base = self.OBS_ROBOT_STATE_SIZE + i * self.OBS_OBSTACLE_DATA_SIZE
            obs_low[base:base+2] = 0.0
            obs_high[base:base+2] = [width, height]
            obs_low[base+2] = 0
            obs_high[base+2] = 10
            obs_low[base+6] = 0
            obs_high[base+6] = 1
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.action_space = spaces.Box(low=np.array([0, -np.pi/4]), high=np.array([20, np.pi/4]), dtype=np.float32)

        self.fig = None
        self.ax = None
        self.robot_patch = None
        self.goal_patch = None
        self.goal_text = None
        self.obstacle_patches = []
        self.path = []
        self.direction_arrow = None
        self.path_line = None
        self.planned_path_line = None
        self.sensor_circle = None
        self.rrt_tree_lines = []
        self.ray_lines = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        margin = self.robot_radius + 10
        self.robot_x = self.np_random.uniform(margin, self.width - margin)
        self.robot_y = self.np_random.uniform(margin, self.height - margin)
        self.robot_orientation = self.np_random.uniform(-np.pi, np.pi)
        self.robot_velocity = 0

        min_start_goal_dist = 250
        while True:
            self.goal_x = self.np_random.uniform(margin, self.width - margin)
            self.goal_y = self.np_random.uniform(margin, self.height - margin)
            if np.sqrt((self.goal_x - self.robot_x)**2 + (self.goal_y - self.robot_y)**2) >= min_start_goal_dist:
                break

        self.obstacles = []
        if self.ax:
            for patch in self.obstacle_patches:
                if patch and hasattr(patch, 'remove'):
                    try:
                        patch.remove()
                    except NotImplementedError:
                        pass
            for artist in list(self.ax.artists):
                try:
                    artist.remove()
                except NotImplementedError:
                    pass
            self.obstacle_patches = []

        num_obstacles = self.np_random.integers(5, self.max_obstacles_in_observation + 1)
        print(f"Generating {num_obstacles} obstacles...")

        dynamic_count = 0
        max_dynamic = int(num_obstacles * 0.3)

        for i in range(num_obstacles):
            attempts = 0
            obstacle_added = False
            while attempts < 100 and not obstacle_added:
                attempts += 1

                shape_type = self.np_random.choice(['circle', 'rectangle'])
                obs_x = self.np_random.uniform(margin, self.width - margin)
                obs_y = self.np_random.uniform(margin, self.height - margin)

                if shape_type == 'circle':
                    radius = self.np_random.uniform(15, 40)
                    shape = Circle(radius)
                    placement_radius = radius
                elif shape_type == 'rectangle':
                    width = self.np_random.uniform(20, 60)
                    height = self.np_random.uniform(20, 60)
                    angle = self.np_random.uniform(-np.pi/4, np.pi/4)
                    shape = Rectangle(width, height, angle)
                    placement_radius = 0.5 * math.sqrt(width**2 + height**2)
                else:
                    continue

                dist_to_start = np.sqrt((obs_x - self.robot_x)**2 + (obs_y - self.robot_y)**2)
                dist_to_goal = np.sqrt((obs_x - self.goal_x)**2 + (obs_y - self.goal_y)**2)
                clear_of_others = True
                for existing_obs in self.obstacles:
                    existing_placement_radius = 0
                    if isinstance(existing_obs.shape, Circle):
                        existing_placement_radius = existing_obs.shape.radius
                    elif isinstance(existing_obs.shape, Rectangle):
                        existing_placement_radius = 0.5 * math.sqrt(existing_obs.shape.width**2 + existing_obs.shape.height**2)
                    dist_to_existing = np.sqrt((obs_x - existing_obs.x)**2 + (obs_y - existing_obs.y)**2)
                    if dist_to_existing < placement_radius + existing_placement_radius + self.robot_radius:
                        clear_of_others = False
                        break

                if (dist_to_start > placement_radius + self.robot_radius + 10 and
                    dist_to_goal > placement_radius + self.robot_radius + 10 and
                    clear_of_others):
                    if not self.available_image_paths:
                        print("Warning: No images available, defaulting to static obstacle.")
                        is_dynamic_classified = False
                        selected_image_path = None
                        label = "no_image"
                    else:
                        selected_image_path = random.choice(self.available_image_paths)
                        is_dynamic_classified = False
                        label = "unknown"

                        if self.classifier:
                            try:
                                _, label, is_dynamic_classified = self.classifier.predict(selected_image_path)
                                print(f"  Obstacle {i+1} @({obs_x:.1f},{obs_y:.1f}): '{os.path.basename(selected_image_path)}' -> '{label}' ({'Dynamic' if is_dynamic_classified else 'Static'})")
                            except Exception as e:
                                print(f"  Warning: Classifier failed for image {os.path.basename(selected_image_path)}: {e}. Defaulting to static.")
                                is_dynamic_classified = False
                                label = "classification_failed"
                        else:
                            print(f"  Warning: Classifier not available for obstacle {i+1}. Using random type.")
                            is_dynamic_classified = (self.np_random.random() < 0.3 and dynamic_count < max_dynamic)

                    if is_dynamic_classified and dynamic_count >= max_dynamic:
                        is_dynamic_classified = False
                        print(f"  Obstacle {i+1} forced to Static to maintain 30% dynamic limit.")

                    if is_dynamic_classified:
                        dynamic_count += 1
                        obs_velocity = self.np_random.uniform(5.0, 15.0)
                        angle = self.np_random.uniform(0, 2*np.pi)
                        obs_direction = np.array([np.cos(angle), np.sin(angle)])
                        obstacle = DynamicObstacle(obs_x, obs_y, shape, selected_image_path, obs_velocity, obs_direction)
                    else:
                        obstacle = StaticObstacle(obs_x, obs_y, shape, selected_image_path)

                    self.obstacles.append(obstacle)
                    obstacle_added = True

            if attempts >= 100 and not obstacle_added:
                print(f"Warning: Could not place obstacle {i+1}.")

        static_count = sum(1 for obs in self.obstacles if isinstance(obs, StaticObstacle))
        print(f"Created {static_count} static and {dynamic_count} dynamic obstacles")

        goal_margin = 1.0
        for obs in self.obstacles:
            if obs.check_collision(self.goal_x, self.goal_y, goal_margin):
                print("Warning: Goal inside obstacle, moving goal.")
                vec_from_obs = np.array([self.goal_x - obs.x, self.goal_y - obs.y])
                dist = np.linalg.norm(vec_from_obs)
                move_dist = 10
                if isinstance(obs.shape, Circle):
                    move_dist = obs.shape.radius + 5
                elif isinstance(obs.shape, Rectangle):
                    move_dist = max(obs.shape.width, obs.shape.height)/2 + 5
                if dist < 1e-6:
                    self.goal_x += move_dist
                else:
                    self.goal_x += vec_from_obs[0] / dist * move_dist
                    self.goal_y += vec_from_obs[1] / dist * move_dist
                self.goal_x = np.clip(self.goal_x, margin, self.width - margin)
                self.goal_y = np.clip(self.goal_y, margin, self.height - margin)

        self.path = [(self.robot_x, self.robot_y)]
        observation = self._get_observation()
        info = self._get_info()

        if self.ax:
            if self.robot_patch:
                self.robot_patch.remove()
                self.robot_patch = None
            if self.goal_patch:
                self.goal_patch.remove()
                self.goal_patch = None
            if self.goal_text:
                self.goal_text.remove()
                self.goal_text = None
            if self.direction_arrow:
                self.direction_arrow.remove()
                self.direction_arrow = None
            if self.path_line:
                self.path_line.remove()
                self.path_line = None
            if self.planned_path_line:
                self.planned_path_line.remove()
                self.planned_path_line = None
            if self.sensor_circle:
                self.sensor_circle.remove()
                self.sensor_circle = None
            for line in self.ray_lines:
                line.remove()
            self.ray_lines = []
            for line in self.rrt_tree_lines:
                line.remove()
            self.rrt_tree_lines = []

        return observation, info

    def _get_observation(self):
        base_obs = [self.robot_x, self.robot_y, self.robot_orientation, self.goal_x, self.goal_y]

        obstacles_with_distance = []
        for obstacle in self.obstacles:
            dist_to_robot_center = np.sqrt((obstacle.x - self.robot_x)**2 + (obstacle.y - self.robot_y)**2)
            approx_obs_radius = 0
            if isinstance(obstacle.shape, Circle):
                approx_obs_radius = obstacle.shape.radius
            elif isinstance(obstacle.shape, Rectangle):
                approx_obs_radius = 0.5 * math.sqrt(obstacle.shape.width**2 + obstacle.shape.height**2)
            effective_dist = dist_to_robot_center - approx_obs_radius
            if effective_dist <= self.sensor_range:
                obstacles_with_distance.append((obstacle, dist_to_robot_center))
                print(f"Obstacle @({obstacle.x:.1f},{obstacle.y:.1f}) within sensor range: dist={dist_to_robot_center:.1f}, effective={effective_dist:.1f}")
            else:
                print(f"Obstacle @({obstacle.x:.1f},{obstacle.y:.1f}) outside sensor range: dist={dist_to_robot_center:.1f}, effective={effective_dist:.1f}")

        obstacles_with_distance.sort(key=lambda x: x[1])

        sensed_obstacles_data = []
        count = 0
        for obstacle, dist in obstacles_with_distance[:self.max_obstacles_in_observation]:
            obs_data = obstacle.get_observation_data()
            sensed_obstacles_data.extend(obs_data)
            count += 1
            print(f"  Sensed obstacle @({obstacle.x:.1f},{obstacle.y:.1f}): Type={'Dynamic' if obstacle.type == ObstacleType.DYNAMIC else 'Static'}, Shape={type(obstacle.shape).__name__}, Vel=({obs_data[7]:.1f},{obs_data[8]:.1f}), Dist={dist:.1f}")

        num_missing = self.max_obstacles_in_observation - count
        if num_missing > 0:
            sensed_obstacles_data.extend([0.0] * self.OBS_OBSTACLE_DATA_SIZE * num_missing)

        observation = np.array(base_obs + sensed_obstacles_data, dtype=np.float32)

        if observation.shape[0] != self.observation_space.shape[0]:
            print(f"FATAL Error: Observation shape mismatch. Got {observation.shape}, expected {self.observation_space.shape}")
            expected_len = self.observation_space.shape[0]
            current_len = len(observation)
            if current_len > expected_len:
                observation = observation[:expected_len]
            elif current_len < expected_len:
                observation = np.pad(observation, (0, expected_len - current_len), 'constant')
            print(f"Fixed observation shape to {observation.shape}")

        return observation

    def _get_info(self):
        current_dist_to_goal = np.sqrt((self.robot_x - self.goal_x)**2 + (self.robot_y - self.robot_y)**2)
        return {
            "distance_to_goal": current_dist_to_goal,
            "ground_truth_obstacles": self.obstacles
        }

    def step(self, action):
        self.current_step += 1
        prev_distance = np.sqrt((self.robot_x - self.goal_x)**2 + (self.robot_y - self.goal_y)**2)

        velocity, steering_angle = action
        velocity = np.clip(velocity, self.action_space.low[0], self.action_space.high[0])
        steering_angle = np.clip(steering_angle, self.action_space.low[1], self.action_space.high[1])

        # Check WaitingRule for dynamic obstacles
        dynamic_obstacles = [obs for obs in self.obstacles if isinstance(obs, DynamicObstacle)]
        predicted_collisions = self.waiting_rule.check_dynamic_collisions(
            robot_x=self.robot_x,
            robot_y=self.robot_y,
            robot_velocity=velocity,
            robot_orientation=self.robot_orientation,
            dynamic_obstacles=dynamic_obstacles
        )
        if self.waiting_rule.should_wait(predicted_collisions):
            print(f"Waiting due to predicted collision with dynamic obstacle(s)")
            velocity = 0  # Pause robot

        dt = 0.5
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

        if not (self.robot_radius <= new_x <= self.width - self.robot_radius and
                self.robot_radius <= new_y <= self.height - self.robot_radius):
            reward = -50
            terminated = True
            info['status'] = 'boundary_collision'
            observation = self._get_observation()
            info.update(self._get_info())
            return observation, reward, terminated, truncated, info

        collision = False
        colliding_obs_type_name = "Unknown"
        for obstacle in self.obstacles:
            if obstacle.check_collision(new_x, new_y, self.robot_radius):
                collision = True
                colliding_obs_type_name = type(obstacle.shape).__name__
                break

        if collision:
            reward = -50
            terminated = True
            info['status'] = f'obstacle_collision ({colliding_obs_type_name})'
            observation = self._get_observation()
            info.update(self._get_info())
            return observation, reward, terminated, truncated, info

        self.robot_x = new_x
        self.robot_y = new_y
        self.path.append((self.robot_x, self.robot_y))

        bounds = (0, 0, self.width, self.height)
        for obstacle in self.obstacles:
            if isinstance(obstacle, DynamicObstacle):
                obstacle.update(dt=dt, bounds=bounds)
                print(f"Updated DynamicObstacle @({obstacle.x:.1f},{obstacle.y:.1f})")
            else:
                print(f"Skipped update for StaticObstacle @({obstacle.x:.1f},{obstacle.y:.1f})")

        distance_to_goal = np.sqrt((self.robot_x - self.goal_x)**2 + (self.robot_y - self.goal_y)**2)
        goal_threshold = self.robot_radius + 5
        if distance_to_goal < goal_threshold:
            reward = 200
            terminated = True
            info['status'] = 'goal_reached'
        else:
            reward_dist = prev_distance - distance_to_goal
            reward_time = -0.5
            reward = (reward_dist * 1.5) + reward_time
            if self.current_step >= self.max_steps:
                truncated = True
                reward -= 50
                info['status'] = 'max_steps_reached'

        observation = self._get_observation()
        info.update(self._get_info())
        return observation, reward, terminated, truncated, info

    def render(self, mode='human', controller_info=None):
        if mode not in self.metadata['render.modes']:
            raise ValueError(f"Unsupported render mode: {mode}")

        if self.fig is None:
            if mode == 'human':
                plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.ax.set_aspect('equal')
            self.ax.grid(True)
            plt.title("Indoor Robot Simulation (Image Obstacles)")
            plt.xlabel("X")
            plt.ylabel("Y")

        if self.robot_patch:
            self.robot_patch.center = (self.robot_x, self.robot_y)
        else:
            self.robot_patch = patches.Circle((self.robot_x, self.robot_y), self.robot_radius, fc='blue', alpha=0.8, zorder=5)
            self.ax.add_patch(self.robot_patch)

        if self.direction_arrow:
            self.direction_arrow.remove()
        arrow_len = self.robot_radius * 1.5
        end_x = self.robot_x + arrow_len * np.cos(self.robot_orientation)
        end_y = self.robot_y + arrow_len * np.sin(self.robot_orientation)
        self.direction_arrow = self.ax.arrow(self.robot_x, self.robot_y, end_x - self.robot_x, end_y - self.robot_y,
                                            head_width=max(self.robot_radius * 0.4, 3), head_length=max(self.robot_radius * 0.6, 5),
                                            fc='red', ec='red', length_includes_head=True, zorder=6)

        if self.goal_patch is None:
            self.goal_patch = patches.Circle((self.goal_x, self.goal_y), self.robot_radius * 0.8, fc='lime', alpha=0.8, ec='green', lw=2, zorder=4)
            self.ax.add_patch(self.goal_patch)
            self.goal_text = self.ax.text(self.goal_x, self.goal_y, 'G', ha='center', va='center', color='black', weight='bold', zorder=5)

        # Clear existing obstacle patches and annotations
        for patch in self.obstacle_patches:
            if patch and hasattr(patch, 'remove'):
                try:
                    patch.remove()
                except NotImplementedError:
                    pass
        for artist in list(self.ax.artists):
            try:
                artist.remove()
            except NotImplementedError:
                pass
        self.obstacle_patches = []

        # Add new obstacle patches
        new_obstacle_patches = []
        for obstacle in self.obstacles:
            patch_artist = obstacle.get_render_patch(self.ax, alpha=0.6, zorder=3)
            if isinstance(patch_artist, patches.Patch) and patch_artist not in self.ax.patches:
                self.ax.add_patch(patch_artist)
            elif isinstance(patch_artist, AnnotationBbox):
                self.ax.add_artist(patch_artist)
            new_obstacle_patches.append(patch_artist)
        self.obstacle_patches = new_obstacle_patches

        if self.path:
            path_x, path_y = zip(*self.path)
            if self.path_line:
                self.path_line.set_data(path_x, path_y)
            else:
                self.path_line, = self.ax.plot(path_x, path_y, 'b-', linewidth=1.5, alpha=0.6, label='Robot Path', zorder=2)

        # Render rays from RayTracingAlgorithm
        for line in self.ray_lines:
            line.remove()
        self.ray_lines = []
        _, ray_viz_points = self.ray_tracer.trace_rays(
            robot_x=self.robot_x,
            robot_y=self.robot_y,
            robot_orientation=self.robot_orientation,
            obstacles=self.obstacles
        )
        for (start_x, start_y), (end_x, end_y) in ray_viz_points:
            line, = self.ax.plot([start_x, end_x], [start_y, end_y], 'y-', alpha=0.3, zorder=1)
            self.ray_lines.append(line)

        for line in self.rrt_tree_lines:
            line.remove()
        self.rrt_tree_lines = []
        if self.planned_path_line:
            self.planned_path_line.remove()
            self.planned_path_line = None

        if controller_info:
            if 'rrt_nodes' in controller_info and 'rrt_parents' in controller_info and controller_info['rrt_nodes'] is not None:
                nodes, parents = controller_info['rrt_nodes'], controller_info['rrt_parents']
                for i, p_idx in enumerate(parents):
                    if p_idx != -1 and i < len(nodes) and p_idx < len(nodes):
                        line, = self.ax.plot([nodes[i][0], nodes[p_idx][0]], [nodes[i][1], nodes[p_idx][1]],
                                            'grey', alpha=0.3, linewidth=0.5, zorder=1)
                        self.rrt_tree_lines.append(line)
            if 'planned_path' in controller_info and controller_info['planned_path']:
                path_points = controller_info['planned_path']
                if len(path_points) > 1:
                    path_x, path_y = zip(*path_points)
                    self.planned_path_line, = self.ax.plot(path_x, path_y, 'g--', linewidth=2, alpha=0.7, label='Planned Path', zorder=4)

        if self.sensor_circle:
            self.sensor_circle.center = (self.robot_x, self.robot_y)
        else:
            self.sensor_circle = patches.Circle((self.robot_x, self.robot_y), self.sensor_range, fc='none', ec='purple', ls=':', alpha=0.5, label='Sensor Range', zorder=2)
            self.ax.add_patch(self.sensor_circle)

        handles, labels = self.ax.get_legend_handles_labels()
        unique_labels_handles = {}
        for h, l in zip(handles, labels):
            if l and l not in unique_labels_handles:
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