# --- START OF FILE indoor_robot_controller.py ---

# indoor_robot_controller.py

import time
import math
import numpy as np
from algorithms.robot_env.indoor_robot_env import IndoorRobotEnv # Assuming this exists
from components.obstacle import Obstacle, ObstacleType, StaticObstacle, DynamicObstacle # Import specific types if needed
from components.shape import Circle, Rectangle, Shape
from utils.ray_tracing_algorithm import RayTracingAlgorithm
from utils.rrt_planner import RRTPathPlanner
# Uncomment WaitingRule import
from utils.waiting_rule import WaitingRule
from utils.obstacle_identifier import ObstacleIdentifier
import pickle # Import pickle for saving data



# --- Comment out ORCA Imports ---
# from pyorca import Agent as OrcaAgent # Rename to avoid conflict if needed
# from pyorca import orca as calculate_orca_velocity
# from halfplaneintersect import InfeasibleError
# --- End ORCA Imports ---

# ... rest of the imports ...

class IndoorRobotController:
    def __init__(self, env: IndoorRobotEnv): # Type hint env
        # ... existing initializations ...
        self.width = env.width
        self.height = env.height
        self.robot_radius = env.robot_radius
        self.action_space = env.action_space
        # self.observation_space = env.observation_space # Pass to identifier
        self.env_dt = env.dt # Store environment timestep if available

        # Initialize components
        self.obstacle_identifier = ObstacleIdentifier(env.max_obstacles_in_observation)
        self.ray_tracer = RayTracingAlgorithm(self.width, self.height, self.robot_radius, env.sensor_range)
        # Uncomment WaitingRule initialization
        self.waiting_rule = WaitingRule(self.robot_radius, prediction_horizon=8, safety_margin=self.robot_radius*3, time_step=self.env_dt)
        self.path_planner = RRTPathPlanner(self.width, self.height, self.robot_radius)

        # Controller parameters (mostly same)
        self.goal_threshold = self.robot_radius + 5
        self.max_velocity = self.action_space.high[0]
        self.min_velocity = 0.0 # Keep minimum linear velocity? ORCA might output zero.
        self.obstacle_slow_down_distance = self.robot_radius * 5
        self.obstacle_avoid_distance = self.robot_radius * 4 # Used for static avoidance blending
        
        # Replace pure pursuit lookahead with current target index
        self.current_target_index = 0
        self.waypoint_threshold = self.robot_radius * 3  # Distance to consider a waypoint reached
        
        self.path_invalidation_check_horizon = 5

        # --- Remove ORCA Parameters, add WaitingRule state ---
        # self.orca_tau = 2.0 # Prediction horizon for ORCA (tune this)
        # --- End ORCA Parameters ---
        self.is_waiting = False # Uncomment for WaitingRule state

        # Controller state
        self.current_planned_path = None
        self.current_rrt_nodes = None
        self.current_rrt_parents = None
        self.current_path_target_idx = 0
        self.perceived_obstacles = [] # List of Obstacle objects currently seen
        # --- Obstacle Memory ---
        self.predefined_static_obstacles = [] # List of predefined static obstacles that won't be forgotten on reset
        self.discovered_static_obstacles = [] # List of STATIC Obstacle objects remembered
        self.map_obstacles = [] # List of Obstacle objects *used* for the last plan (will be populated from discovered)
        self.known_obstacle_descriptions = set() # Store descriptions for quick lookup
        self.obstacle_memory_tolerance = 0.5 # Positional tolerance for considering an obstacle "known"
        # --- End Obstacle Memory ---
        self.last_replanning_time = -np.inf
        self.status = "Initializing"

        # --- State for Velocity Estimation ---
        # self.last_robot_pos = None
        # self.last_time = None
        # self.current_robot_velocity_vector = np.array([0.0, 0.0]) # Store estimated [vx, vy]
        # self.pre_action = np.array([0.0, 0.0])
        # --- End State ---

        # --- Data Collection ---
        self.collecting_data = False # Flag to enable/disable collection
        self.collected_data = [] # List to store (state, action) pairs
        # -----------------------

    # --- Data Collection Methods ---
    def start_data_collection(self):
        """Enables data collection and clears previous data."""
        self.collecting_data = True
        self.collected_data = []
        print("Data collection started.")

    def stop_data_collection(self):
        """Disables data collection."""
        self.collecting_data = False
        print(f"Data collection stopped. Collected {len(self.collected_data)} samples.")

    def get_collected_data(self):
        """Returns the collected data."""
        return self.collected_data

    def save_collected_data(self, filename="collected_avoidance_data.pkl"):
        """Saves the collected data to a file using pickle."""
        if not self.collected_data:
            print("No data collected to save.")
            return
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.collected_data, f)
            print(f"Collected data saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")
    # -----------------------------


    def reset(self):
         # ... existing reset logic ...
        #  self.last_robot_pos = None
        #  self.last_time = None
        #  self.current_robot_velocity_vector = np.array([0.0, 0.0])
         self.current_planned_path = None
         self.current_rrt_nodes = None
         self.current_rrt_parents = None
         self.current_path_target_idx = 0
         self.current_target_index = 0
         self.perceived_obstacles = []
         # --- Reset Obstacle Memory ---
         self.discovered_static_obstacles = [] # Clear discovered obstacles
         self.map_obstacles = []
         self.known_obstacle_descriptions = set()
         # --- End Reset Obstacle Memory ---
         self.is_waiting = False  # Reset waiting state
         self.last_replanning_time = -np.inf
         self.status = "Reset"
        #  self.pre_action = np.array([0.0, 0.0])
         # Do NOT reset collected_data here
         
         # Re-add any predefined static obstacles
         if self.predefined_static_obstacles:
             self.add_static_obstacles(self.predefined_static_obstacles)
             
    def add_static_obstacles(self, obstacles):
        """
        Add static obstacles to the controller's known obstacles list.
        Args:
            obstacles: A list of Obstacle objects of type STATIC
        Returns:
            int: Number of valid static obstacles added
        """
        added_count = 0
        for obs in obstacles:
            if obs.type == ObstacleType.STATIC:
                desc = obs.get_static_description()
                if desc not in self.known_obstacle_descriptions:
                    self.discovered_static_obstacles.append(obs)
                    self.known_obstacle_descriptions.add(desc)
                    # Also store in predefined obstacles so they're preserved on reset
                    if obs not in self.predefined_static_obstacles:
                        self.predefined_static_obstacles.append(obs)
                    added_count += 1
        return added_count

    # --- Obstacle Memory Helper ---
    def _update_obstacle_memory(self, currently_perceived_obstacles: list):
        """
        Updates the list of discovered static obstacles based on current perception.
        Only adds newly seen static obstacles.
        """
        newly_discovered_count = 0
        for obs in currently_perceived_obstacles:
            if obs.type == ObstacleType.STATIC:
                # Use the static description (initial pos, shape) for uniqueness
                desc = obs.get_static_description()
                is_known = False
                # Check if an obstacle with a very similar description exists
                # Using a set of descriptions is faster than iterating and comparing distances
                if desc in self.known_obstacle_descriptions:
                     is_known = True
                else:
                    # If not found by exact description, do a tolerance check (optional, adds robustness)
                    # This is slower but handles slight init variations if descriptions aren't perfect
                    # for known_obs in self.discovered_static_obstacles:
                    #     known_desc = known_obs.get_static_description()
                    #     # Compare position with tolerance, and shape info exactly
                    #     if (np.linalg.norm(np.array(desc[:2]) - np.array(known_desc[:2])) < self.obstacle_memory_tolerance and
                    #         desc[2:] == known_desc[2:]):
                    #         is_known = True
                    #         # Optimization: Add the current description to the set if found via tolerance
                    #         self.known_obstacle_descriptions.add(desc)
                    #         break
                    pass # Keep is_known = False if not in set

                if not is_known:
                    # print(f"Discovered new static obstacle: {desc}") # Debug print
                    self.discovered_static_obstacles.append(obs)
                    self.known_obstacle_descriptions.add(desc) # Add its description to the set
                    newly_discovered_count += 1
        # Return True if the map changed significantly (might trigger replan)
        return newly_discovered_count > 0


    # --- Helper to get effective radius for ORCA ---
    def _get_orca_effective_radius(self, shape: Shape) -> float:
        """ Returns an effective radius for shapes for use with ORCA. """
        padding = 0.5 * self.robot_radius # Padding for ORCA
        return shape.get_effective_radius() + padding

    # --- Helper to convert action to velocity vector ---
    def _action_to_velocity_vector(self, linear_velocity: float, steering_angle: float, robot_orientation: float, robot_x: float, robot_y: float) -> np.ndarray:
        """ Converts [linear_vel, steering_angle] action relative to robot to world velocity [vx, vy]. """
        # Use the intended linear velocity in the *current* robot orientation
        # ORCA works best with a preferred velocity vector. Pointing towards the target point is better.
        target_point = self._get_current_target_point(robot_x, robot_y) # Use last known pos

        if target_point is None or not self.current_planned_path:
             # Fallback: Use current orientation or stop if no path
             pref_vx = linear_velocity * np.cos(robot_orientation)
             pref_vy = linear_velocity * np.sin(robot_orientation)
        else:
             target_vec = np.array(target_point) - np.array([robot_x, robot_y])
             dist = np.linalg.norm(target_vec)
             if dist < 1e-6:
                  # Very close to target, use current orientation or aim for next point if possible
                  pref_vx = linear_velocity * np.cos(robot_orientation)
                  pref_vy = linear_velocity * np.sin(robot_orientation)
             else:
                  # Point the preferred velocity towards target, scaled by intended linear speed
                  unit_target_vec = target_vec / dist
                  pref_vx = linear_velocity * unit_target_vec[0]
                  pref_vy = linear_velocity * unit_target_vec[1]

        # Limit preferred speed by max_velocity
        pref_vel_vec = np.array([pref_vx, pref_vy])
        speed = np.linalg.norm(pref_vel_vec)
        if speed > self.max_velocity:
            pref_vel_vec = (pref_vel_vec / speed) * self.max_velocity

        return pref_vel_vec

    # --- Helper to convert velocity vector back to action ---
    def _velocity_vector_to_action(self, optimal_velocity_vector: np.ndarray, robot_orientation: float) -> np.ndarray:
        """ Converts world velocity [vx, vy] back to robot action [linear_vel, steering_angle]. """
        linear_velocity = np.linalg.norm(optimal_velocity_vector)

        if linear_velocity < 1e-6:
            # If stopping, maintain current orientation (zero steering)
            steering_angle = 0.0
            linear_velocity = 0.0 # Ensure zero linear velocity too
        else:
            target_orientation = np.arctan2(optimal_velocity_vector[1], optimal_velocity_vector[0])
            # Calculate required change in orientation
            steering_angle = self._normalize_angle(target_orientation - robot_orientation)

            # Consider turn rate limits implicitly via action space clipping
            # No, steering angle IS the desired angle relative to current.
            # It's not a rate, it's the target offset. The environment/robot dynamics handle how quickly it turns.

        # Clip action based on action space limits
        linear_velocity = np.clip(linear_velocity, 0.0, self.action_space.high[0]) # Ensure non-negative velocity
        steering_angle = np.clip(steering_angle, self.action_space.low[1], self.action_space.high[1])

        return np.array([linear_velocity, steering_angle], dtype=np.float32)

    # --- NEW: Get current target point from RRT path ---
    def _get_current_target_point(self, robot_x, robot_y):
        """Get the current target point along the RRT path."""
        if not self.current_planned_path or len(self.current_planned_path) == 0:
            return None
            
        # Make sure target index is within bounds
        if self.current_target_index >= len(self.current_planned_path):
            self.current_target_index = len(self.current_planned_path) - 1
            
        # Get current target point
        current_target = self.current_planned_path[self.current_target_index]
        
        # Check if we've reached the current target
        robot_pos = np.array([robot_x, robot_y])
        target_pos = np.array(current_target)
        distance_to_target = np.linalg.norm(robot_pos - target_pos)
        
        # If we're getting close to the current target, start looking ahead to the next waypoint
        # for smoother transitions
        if self.current_target_index < len(self.current_planned_path) - 1:
            look_ahead_distance = max(self.waypoint_threshold * 2, self.robot_radius * 3)
            
            if distance_to_target < look_ahead_distance:
                # Calculate a blended target point between current and next waypoint
                next_target = self.current_planned_path[self.current_target_index + 1]
                next_target_pos = np.array(next_target)
                
                # Blend factor depends on how close we are to current waypoint
                blend_factor = 1.0 - (distance_to_target / look_ahead_distance)
                blend_factor = min(0.8, blend_factor)  # Cap at 0.8 to avoid jumping too quickly
                
                # Blend current and next target
                blended_target = target_pos * (1 - blend_factor) + next_target_pos * blend_factor
                
                # If very close to current waypoint, move to next one
                if distance_to_target < self.waypoint_threshold:
                    self.current_target_index += 1
                
                return blended_target.tolist()
        
        # If we've reached the current target, move to the next one
        if distance_to_target < self.waypoint_threshold and self.current_target_index < len(self.current_planned_path) - 1:
            self.current_target_index += 1
            current_target = self.current_planned_path[self.current_target_index]
            
        return current_target

    # --- NEW: Calculate RRT path following action ---
    def _calculate_rrt_following_action(self, robot_x, robot_y, robot_orientation, perceived_static_obstacles):
        """Calculate action to follow the RRT-planned path directly."""
        # Returns: action = np.array([velocity, steering_angle]), calculation_status
        
        calculation_status = "Following RRT Path"
        if not self.current_planned_path or len(self.current_planned_path) == 0:
            calculation_status = "Path Error (No Path) - Stopping"
            return np.array([0.0, 0.0]), calculation_status
            
        # Get current target point
        target_point = self._get_current_target_point(robot_x, robot_y)
        if target_point is None:
            calculation_status = "Path Error (No Target) - Stopping"
            return np.array([0.0, 0.0]), calculation_status
            
        # Calculate direction to target
        target_x, target_y = target_point
        target_vector = np.array([target_x - robot_x, target_y - robot_y])
        target_distance = np.linalg.norm(target_vector)
        
        if target_distance < 1e-6:
            # We're at the target point, check if there's a next point
            if self.current_target_index < len(self.current_planned_path) - 1:
                self.current_target_index += 1
                target_point = self.current_planned_path[self.current_target_index]
                target_x, target_y = target_point
                target_vector = np.array([target_x - robot_x, target_y - robot_y])
                target_distance = np.linalg.norm(target_vector)
                if target_distance < 1e-6:
                    # Still too close, just use robot's current orientation
                    target_orientation = robot_orientation
                else:
                    target_orientation = np.arctan2(target_vector[1], target_vector[0])
            else:
                # We're at the last point, use robot's current orientation
                target_orientation = robot_orientation
        else:
            target_orientation = np.arctan2(target_vector[1], target_vector[0])
            
        # --- Reactive Static Obstacle Avoidance ---
        final_target_orientation = target_orientation
        min_dist_to_static_boundary = float('inf')
        avoiding_obstacle = None
        current_min_avoid_eff_dist = float('inf')

        # Kiểm tra vật cản tĩnh với WaitingRule trước
        if self.waiting_rule.check_static_collision(robot_x, robot_y, self.max_velocity, robot_orientation, perceived_static_obstacles):
            # Nếu có vật cản tĩnh phía trước, dừng lại
            return np.array([0.0, 0.0]), "Emergency stop - Static obstacle detected ahead"

        for obstacle in perceived_static_obstacles:
            obs_pos = np.array(obstacle.get_position())
            robot_pos = np.array([robot_x, robot_y])

            eff_vec = obstacle.shape.get_effective_vector(robot_x, robot_y, obs_pos[0], obs_pos[1])
            eff_dist = np.linalg.norm(eff_vec) # Effective distance
            min_dist_to_static_boundary = min(min_dist_to_static_boundary, eff_dist)

            if eff_dist < self.obstacle_avoid_distance:
                vec_to_obs = eff_vec
                # Check if obstacle is roughly in front based on target direction
                angle_to_obs = np.arctan2(vec_to_obs[1], vec_to_obs[0])
                angle_diff_to_target = abs(self._normalize_angle(angle_to_obs - target_orientation))

                # Avoidance cone - check all obstacles
                if angle_diff_to_target < np.pi * 2:
                    # Choose the closest one within the cone based on eff_dist
                    if avoiding_obstacle is None or eff_dist < current_min_avoid_eff_dist:
                            avoiding_obstacle = obstacle
                            current_min_avoid_eff_dist = eff_dist # Update min distance for chosen one

        if avoiding_obstacle is not None:
            final_avoid_eff_dist = current_min_avoid_eff_dist
            calculation_status = f"Avoiding static {type(avoiding_obstacle.shape).__name__}"
            
            # Use ray tracing for avoidance direction
            avoidance_orientation = self.ray_tracer.avoid_static_obstacle(
                robot_x, robot_y, self.robot_radius, robot_orientation, avoiding_obstacle, target_x, target_y
            )
            
            # Blend factor based on proximity
            blend_factor = 1.0 - np.clip(final_avoid_eff_dist / self.obstacle_avoid_distance, 0.0, 1.0) ** 4
            
            final_target_orientation = self._slerp_angle(target_orientation, avoidance_orientation, blend_factor)
            
        # --- Calculate Control Action ---
        # Tính góc lái mượt hơn bằng cách giới hạn sự thay đổi giữa các bước
        # Tạo góc chuyển hướng mượt mà hơn
        # Lưu góc lái trước đó nếu chưa có
        if not hasattr(self, 'prev_steering_angle'):
            self.prev_steering_angle = 0.0
            
        desired_steering_angle = self._normalize_angle(final_target_orientation - robot_orientation)
        
        # Giới hạn sự thay đổi góc lái giữa các bước
        max_steering_change = 0.3  # Giới hạn thay đổi góc lái mỗi bước (~17 độ)
        steering_change = desired_steering_angle - self.prev_steering_angle
        steering_change = np.clip(steering_change, -max_steering_change, max_steering_change)
        
        # Cập nhật góc lái
        steering_angle = self.prev_steering_angle + steering_change
        steering_angle = np.clip(steering_angle, self.action_space.low[1], self.action_space.high[1])
        
        # Lưu lại góc lái hiện tại cho bước tiếp theo
        self.prev_steering_angle = steering_angle

        # --- Calculate velocity based on steering and obstacle proximity ---
        velocity = self.max_velocity
        
        # Factor based on steering magnitude
        max_steer = abs(self.action_space.high[1]) if self.action_space.high[1] > 1e-6 else np.pi # Avoid division by zero
        norm_steer_mag = abs(steering_angle) / max_steer
        
        # Làm mượt hơn khi chuyển hướng
        steering_vel_factor = max(0.3, np.cos(norm_steer_mag * np.pi / 2))

        # Factor based on proximity to closest PERCEIVED static obstacle boundary
        proximity_vel_factor = 1.0
        if min_dist_to_static_boundary < self.obstacle_avoid_distance:
             proximity_vel_factor = np.clip(min_dist_to_static_boundary / self.obstacle_avoid_distance, 0.2, 1.0)
        elif min_dist_to_static_boundary < self.obstacle_slow_down_distance:
            slow_down_ratio = (min_dist_to_static_boundary - self.obstacle_avoid_distance) / (self.obstacle_slow_down_distance - self.obstacle_avoid_distance)
            proximity_vel_factor = 0.6 + 0.4 * slow_down_ratio
            proximity_vel_factor = np.clip(proximity_vel_factor, 0.6, 1.0)

        # Factor based on distance to waypoint - slow down when approaching waypoints
        waypoint_vel_factor = 1.0
        if target_distance < self.waypoint_threshold * 3:
            waypoint_vel_factor = np.clip(target_distance / (self.waypoint_threshold * 3), 0.4, 1.0)
            
        velocity = self.max_velocity * steering_vel_factor * proximity_vel_factor * waypoint_vel_factor
        
        # Lưu vận tốc trước đó nếu chưa có
        if not hasattr(self, 'prev_velocity'):
            self.prev_velocity = velocity
            
        # Làm mượt sự thay đổi vận tốc
        max_velocity_change = 0.1 * self.max_velocity  # Giới hạn thay đổi vận tốc mỗi bước
        velocity_change = velocity - self.prev_velocity
        velocity_change = np.clip(velocity_change, -max_velocity_change, max_velocity_change)
        
        # Cập nhật vận tốc
        velocity = self.prev_velocity + velocity_change
        velocity = np.clip(velocity, self.min_velocity, self.max_velocity)
        
        # Lưu lại vận tốc hiện tại cho bước tiếp theo
        self.prev_velocity = velocity

        action = np.array([velocity, steering_angle], dtype=np.float32)
        return action, calculation_status


    def get_action(self, observation: dict):
        # ------------------------------------------------------
        # STEP 1: OBSTACLE IDENTIFICATION (ưu tiên cao nhất)
        # ------------------------------------------------------
        robot_x, robot_y, robot_velocity, robot_orientation, goal_x, goal_y = observation['base_observation']
        
        # Kiểm tra xem đã đạt mục tiêu chưa
        distance_to_goal = np.linalg.norm(np.array([robot_x, robot_y]) - np.array([goal_x, goal_y]))
        if distance_to_goal < self.goal_threshold:
            self.status = "Goal Reached - Stopping"
            self.current_planned_path = None
            self.current_path_target_idx = 0
            self.current_target_index = 0
            # Reset saved velocities and steering angles
            if hasattr(self, 'prev_velocity'):
                del self.prev_velocity
            if hasattr(self, 'prev_steering_angle'):
                del self.prev_steering_angle
            info = self._get_controller_info()
            info['goal_reached'] = True
            final_action = np.array([0.0, 0.0]) # Ensure stop action
            return final_action, info
            
        # Nhận diện vật cản - bỏ qua phần GoogleNet theo yêu cầu
        self.perceived_obstacles = self.obstacle_identifier.identify(observation)
        
        # Cập nhật bộ nhớ về vật cản tĩnh
        memory_updated = self._update_obstacle_memory(self.perceived_obstacles)
        
        # Phân loại vật cản thành tĩnh và động
        perceived_static_obstacles = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.STATIC]
        perceived_dynamic_obstacles = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.DYNAMIC]

        # ------------------------------------------------------
        # STEP 2: OBSTACLE AVOIDANCE (ưu tiên thứ hai)
        # ------------------------------------------------------
        
        # Khởi tạo biến để theo dõi việc áp dụng các chiến lược tránh vật cản
        static_avoidance_applied = False
        dynamic_avoidance_applied = False
        avoidance_action = np.array([0.0, 0.0])
        avoidance_status = ""
        
        # A. Xử lý vật cản động bằng Waiting Rule
        if perceived_dynamic_obstacles and robot_velocity > 1e-3:
            # Kiểm tra va chạm với vật cản động
            predicted_collisions = self.waiting_rule.check_dynamic_collisions(
                robot_x, robot_y, robot_velocity, robot_orientation, perceived_dynamic_obstacles
            )
            
            # Quyết định có nên chờ không
            should_wait, deceleration = self.waiting_rule.should_wait(predicted_collisions)
            self.is_waiting = should_wait
            
            if self.is_waiting:
                dynamic_avoidance_applied = True
                if deceleration is None:
                    # Dừng khẩn cấp
                    avoidance_action = np.array([0.0, 0.0])
                    avoidance_status = "Emergency Stop for Dynamic Obstacle"
                else:
                    print(f"Deceleration: {deceleration}")
                    # Giảm tốc độ dần dần
                    reduced_velocity = max(0.0, robot_velocity * 0.5)
                    if reduced_velocity < 0.1:
                        reduced_velocity = 0.0
                        avoidance_status = "Stopped for Dynamic Obstacle"
                    else:
                        avoidance_status = f"Decelerating for Dynamic Obstacle: {reduced_velocity:.2f}"
                    
                    # Duy trì hướng hiện tại khi giảm tốc
                    avoidance_action = np.array([reduced_velocity, 0.0])
        else:
            self.is_waiting = False
            
        # B. Xử lý vật cản tĩnh bằng Ray Tracing Algorithm (chỉ khi không áp dụng waiting rule)
        if perceived_static_obstacles and not dynamic_avoidance_applied:
            # Kiểm tra va chạm với vật cản tĩnh
            if self.waiting_rule.check_static_collision(robot_x, robot_y, robot_velocity, robot_orientation, perceived_static_obstacles):
                # Nếu có vật cản tĩnh phía trước gần, dừng lại
                static_avoidance_applied = True
                avoidance_action = np.array([0.0, 0.0])
                avoidance_status = "Emergency stop - Static obstacle detected ahead"
            else:
                # Nếu có đường đi hiện tại, áp dụng ray tracing để tránh vật cản tĩnh
                if self.current_planned_path and len(self.current_planned_path) > 1:
                    # Lấy điểm mục tiêu hiện tại
                    target_point = self._get_current_target_point(robot_x, robot_y)
                    if target_point:
                        target_x, target_y = target_point
                        
                        # Tìm vật cản gần nhất cần tránh
                        closest_obstacle = None
                        min_distance = float('inf')
                        
                        for obstacle in perceived_static_obstacles:
                            obs_pos = np.array(obstacle.get_position())
                            robot_pos = np.array([robot_x, robot_y])
                            eff_vec = obstacle.shape.get_effective_vector(robot_x, robot_y, obs_pos[0], obs_pos[1])
                            eff_dist = np.linalg.norm(eff_vec)
                            
                            if eff_dist < self.obstacle_avoid_distance and eff_dist < min_distance:
                                closest_obstacle = obstacle
                                min_distance = eff_dist
                        
                        if closest_obstacle:
                            static_avoidance_applied = True
                            
                            # Sử dụng ray tracing để tìm hướng tránh vật cản
                            target_orientation = np.arctan2(target_y - robot_y, target_x - robot_x)
                            avoidance_orientation = self.ray_tracer.avoid_static_obstacle(
                                robot_x, robot_y, self.robot_radius, robot_orientation, 
                                closest_obstacle, target_x, target_y
                            )
                            
                            # Tính góc lái để tránh vật cản
                            steering_angle = self._normalize_angle(avoidance_orientation - robot_orientation)
                            steering_angle = np.clip(steering_angle, self.action_space.low[1], self.action_space.high[1])
                            
                            # Giảm tốc độ khi gần vật cản
                            proximity_factor = np.clip(min_distance / self.obstacle_avoid_distance, 0.2, 1.0)
                            reduced_velocity = self.max_velocity * proximity_factor
                            
                            avoidance_action = np.array([reduced_velocity, steering_angle])
                            avoidance_status = f"Avoiding static obstacle with Ray Tracing"

        # ------------------------------------------------------
        # STEP 3: PATH PLANNING (ưu tiên thấp nhất)
        # ------------------------------------------------------
        
        # Chỉ thực hiện lập kế hoạch đường đi nếu không áp dụng các chiến lược tránh vật cản
        if not static_avoidance_applied and not dynamic_avoidance_applied:
            # Kiểm tra xem có cần lập kế hoạch đường đi mới không
            needs_replan = False
            
            if self.current_planned_path is None or len(self.current_planned_path) <= 1:
                needs_replan = True
            elif self.current_target_index >= len(self.current_planned_path):
                needs_replan = True
            elif self._is_path_invalidated(robot_x, robot_y, perceived_dynamic_obstacles + perceived_static_obstacles):
                needs_replan = True
                
            # Lập kế hoạch đường đi mới nếu cần
            if needs_replan:
                self.status = "Planning new path with RRT"
                # Sử dụng vật cản tĩnh đã phát hiện cho việc lập kế hoạch
                self.map_obstacles = list(self.discovered_static_obstacles)
                
                # Đặt mục tiêu
                self.path_planner.set_goal(goal_x, goal_y)
                
                # Lập kế hoạch đường đi bằng RRT
                new_path, nodes, parents = self.path_planner.plan_path(
                    robot_x, robot_y, goal_x, goal_y, self.map_obstacles, smooth_path=False
                )
                
                if new_path and len(new_path) > 1:
                    self.current_planned_path = new_path
                    self.current_rrt_nodes = nodes
                    self.current_rrt_parents = parents
                    self.current_path_target_idx = 0
                    self.current_target_index = 0
                    
                    # Reset các biến theo dõi chuyển động
                    if hasattr(self, 'prev_velocity'):
                        del self.prev_velocity
                    if hasattr(self, 'prev_steering_angle'):
                        del self.prev_steering_angle
                        
                    self.status = "New path planned with RRT"
                else:
                    self.status = "Path planning failed"
                    self.current_planned_path = None
                    self.current_rrt_nodes = nodes
                    self.current_rrt_parents = parents
            
            # Đi theo đường đã lập kế hoạch
            if self.current_planned_path and len(self.current_planned_path) > 1:
                # Tính toán hành động để đi theo đường đã lập kế hoạch
                path_following_action, path_status = self._calculate_rrt_following_action(
                    robot_x, robot_y, robot_orientation, []  # Không xét vật cản tĩnh nữa vì đã xử lý ở trên
                )
                avoidance_action = path_following_action
                avoidance_status = "Following planned path"
            else:
                # Không có đường đi, dừng lại
                avoidance_action = np.array([0.0, 0.0])
                avoidance_status = "No path available - Stopping"
                
        # ------------------------------------------------------
        # STEP 4: SET FINAL ACTION
        # ------------------------------------------------------
        
        # Đặt hành động cuối cùng
        final_action = avoidance_action
        
        # Cập nhật trạng thái
        self.status = avoidance_status
        
        # Trả về thông tin
        controller_info = self._get_controller_info()
        controller_info["status"] = self.status
        controller_info["is_waiting"] = self.is_waiting
        controller_info["discovered_static_obstacles"] = self.discovered_static_obstacles
        controller_info["map_obstacles_used_for_plan"] = self.map_obstacles
        
        return final_action, controller_info


    def _is_path_invalidated(self, robot_x, robot_y, obstacles_to_check: list):
        if not self.current_planned_path or len(self.current_planned_path) < 2:
             return False

        robot_pos = np.array([robot_x, robot_y])

        # --- Find segment closest to robot ---
        min_dist_sq = float('inf')
        closest_segment_idx = 0
        search_center = self.current_target_index  # Use current target index instead of path_target_idx
        search_radius = 5
        search_start = max(0, search_center - search_radius)
        search_end = min(len(self.current_planned_path) - 2, search_center + search_radius)
        search_end = max(search_start, search_end)

        for i in range(search_start, search_end + 1):
             if i >= len(self.current_planned_path) - 1: break
             p1 = np.array(self.current_planned_path[i])
             p2 = np.array(self.current_planned_path[i+1])
             seg_vec = p2 - p1
             seg_len_sq = np.dot(seg_vec, seg_vec)
             if seg_len_sq < 1e-9:
                  dist_sq = np.sum((robot_pos - p1)**2)
             else:
                  t = np.clip(np.dot(robot_pos - p1, seg_vec) / seg_len_sq, 0, 1)
                  closest_point = p1 + t * seg_vec
                  dist_sq = np.sum((robot_pos - closest_point)**2)

             if dist_sq < min_dist_sq:
                  min_dist_sq = dist_sq
                  closest_segment_idx = i

        # Check segments ahead from the closest segment found
        start_check_idx = closest_segment_idx
        end_check_idx = min(len(self.current_planned_path) - 2, start_check_idx + self.path_invalidation_check_horizon)
        start_check_idx = min(start_check_idx, len(self.current_planned_path) - 2)
        if start_check_idx < 0: return False

        for i in range(start_check_idx, end_check_idx + 1):
             if i >= len(self.current_planned_path) - 1: break
             p1_tuple = tuple(self.current_planned_path[i])
             p2_tuple = tuple(self.current_planned_path[i+1])

             # Check this path segment against the provided list of obstacles
             for obs in obstacles_to_check: # Use the passed list
                  if obs.intersects_segment(p1_tuple, p2_tuple, self.robot_radius):
                       return True # Path is blocked ahead

        return False


    def _normalize_angle(self, angle):
        while angle > np.pi: angle -= 2 * np.pi
        while angle < -np.pi: angle += 2 * np.pi
        return angle

    def _slerp_angle(self, a1, a2, t):
        a1 = self._normalize_angle(a1)
        a2 = self._normalize_angle(a2)
        diff = self._normalize_angle(a2 - a1)
        interpolated = self._normalize_angle(a1 + diff * t)
        return interpolated


    def _get_controller_info(self):
        # Make sure to return copies if these lists might be modified elsewhere
        return {
            "status": self.status,
            "planned_path": self.current_planned_path.copy() if self.current_planned_path else None,
            "rrt_nodes": self.current_rrt_nodes, # Usually RRT nodes are immutable points
            "rrt_parents": self.current_rrt_parents, # Dict might be mutable, copy if needed
            "target_idx": self.current_target_index, # Use the current target index for direct following
            "is_waiting": self.is_waiting,  # Add waiting state to info
            # Add obstacle memory info for debugging/visualization
            "discovered_static_obstacles": [obs for obs in self.discovered_static_obstacles], # Return a copy
            "map_obstacles_used_for_plan": [obs for obs in self.map_obstacles], # Return a copy
            }

# --- END OF FILE indoor_robot_controller.py ---