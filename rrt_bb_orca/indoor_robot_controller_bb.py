# --- START OF FILE indoor_robot_controller.py ---

# indoor_robot_controller.py

import time
import math
import numpy as np
from indoor_robot_env import IndoorRobotEnv # Assuming this exists
from components.obstacle import Obstacle, ObstacleType, StaticObstacle, DynamicObstacle # Import specific types if needed
from components.shape import Circle, Rectangle, Shape
# from utils.ray_tracing_algorithm import RayTracingAlgorithm # REMOVED Ray Tracing
from utils.rrt_connect_planner import RRTConnectPathPlanner
from utils.rrt_planner import RRTPathPlanner
# Remove or comment out WaitingRule if ORCA completely replaces its action logic
# from utils.waiting_rule import WaitingRule
from utils.obstacle_identifier import ObstacleIdentifier
from utils.bubble_rebound import BubbleRebound # Import the Bubble Rebound class
import pickle # Import pickle for saving data



# --- ORCA Imports ---
# Adjust path '.' to 'utils.' or similar if you place them in a subdirectory
from pyorca import Agent as OrcaAgent # Rename to avoid conflict if needed
from pyorca import orca as calculate_orca_velocity
from halfplaneintersect import InfeasibleError
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
        # self.ray_tracer = RayTracingAlgorithm(self.width, self.height, self.robot_radius) # REMOVED Ray Tracing
        # self.waiting_rule = WaitingRule(self.robot_radius) # Keep if needed for detection, disable if ORCA handles all
        self.path_planner = RRTConnectPathPlanner(self.width, self.height, self.robot_radius)

        # Controller parameters (mostly same)
        self.goal_threshold = self.robot_radius
        self.max_velocity = self.action_space.high[0]
        self.min_velocity = 0.0 # Keep minimum linear velocity? ORCA might output zero.
        self.obstacle_slow_down_distance = self.robot_radius * 4 # Used for general slowdown based on closest obstacle
        # self.obstacle_avoid_distance = self.robot_radius * 4 # Replaced by dynamic bubble radius for avoidance trigger

        # --- Bubble Rebound Parameters ---
        self.num_ray = 25 
        self.bubble_rebound = BubbleRebound(
            env_width=self.width,
            env_height=self.height,
            env_dt=self.env_dt,
            num_rays=self.num_ray, # Number of rays for bubble calculation
            robot_radius=self.robot_radius,
            sensor_range=env.sensor_range, # Assuming this is the range for bubble calculation
            K_values=[2.5] * self.num_ray, # K values for each ray
        )
        # --- End Bubble Rebound Parameters ---

        # self.lookahead_distance = self.robot_radius * 2 # REMOVED - Replaced by direct waypoint following
        self.waypoint_achieved_threshold = self.robot_radius * 2 # Threshold to consider a waypoint "reached"
        self.path_invalidation_check_horizon = 5

        # --- ORCA Parameters ---
        self.orca_tau = 2.0 # Prediction horizon for ORCA (tune this)
        self.orca_padding_radius = self.max_velocity * self.env_dt * self.orca_tau # Padding radius for ORCA agents
        # --- End ORCA Parameters ---

        # Controller state
        self.current_planned_path = None
        self.current_rrt_nodes = None
        self.current_rrt_parents = None
        self.current_path_target_idx = 0
        self.perceived_obstacles = [] # List of Obstacle objects currently seen
        # --- Obstacle Memory ---
        self.discovered_static_obstacles = [] # List of STATIC Obstacle objects remembered
        self.map_obstacles = [] # List of Obstacle objects *used* for the last plan (will be populated from discovered)
        self.known_obstacle_descriptions = set() # Store descriptions for quick lookup
        self.obstacle_memory_tolerance = 0.5 # Positional tolerance for considering an obstacle "known"
        # --- End Obstacle Memory ---
        # self.is_waiting = False # May not be needed if ORCA handles waits implicitly
        self.last_replanning_time = -np.inf
        self.status = "Initializing"

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
         self.current_planned_path = None
         self.current_rrt_nodes = None
         self.current_rrt_parents = None
         self.current_path_target_idx = 0
         self.perceived_obstacles = []
         self.discovered_static_obstacles = []
         self.map_obstacles = []
         self.known_obstacle_descriptions = set()
         self.last_replanning_time = -np.inf
         self.status = "Reset"
         # Do NOT reset collected_data here

    # --- Obstacle Memory Helper ---
    def _update_obstacle_memory(self, currently_perceived_obstacles: list):
        newly_discovered_count = 0
        for obs in currently_perceived_obstacles:
            if obs.type == ObstacleType.STATIC:
                desc = obs.get_static_description()
                if desc not in self.known_obstacle_descriptions:
                    self.discovered_static_obstacles.append(obs)
                    self.known_obstacle_descriptions.add(desc)
                    newly_discovered_count += 1
        return newly_discovered_count > 0

    # --- Helper to get effective radius for ORCA ---
    def _get_orca_effective_radius(self, shape: Shape) -> float:
        return shape.get_effective_radius() + self.orca_padding_radius

    # --- MODIFIED: Helper to convert action to velocity vector ---
    def _action_to_velocity_vector(self, linear_velocity: float, steering_angle: float, robot_orientation: float) -> np.ndarray:
        """
        Converts [linear_vel, steering_angle] action relative to robot to world velocity [vx, vy].
        The steering_angle is relative to the robot's current orientation and indicates the
        desired change in heading. The linear_velocity is the desired speed in that new heading.
        """
        global_target_orientation = self._normalize_angle(robot_orientation + steering_angle)

        pref_vx = linear_velocity * np.cos(global_target_orientation)
        pref_vy = linear_velocity * np.sin(global_target_orientation)

        pref_vel_vec = np.array([pref_vx, pref_vy])
        speed = np.linalg.norm(pref_vel_vec) 

        if speed > self.max_velocity:
            pref_vel_vec = (pref_vel_vec / speed) * self.max_velocity
        elif speed < self.min_velocity and speed > 1e-6: 
             pass 
        elif speed < 1e-6 : 
             pref_vel_vec = np.array([0.0,0.0])

        return pref_vel_vec

    # --- Helper to convert velocity vector back to action ---
    def _velocity_vector_to_action(self, optimal_velocity_vector: np.ndarray, robot_orientation: float) -> np.ndarray:
        linear_velocity = np.linalg.norm(optimal_velocity_vector)

        if linear_velocity < 1e-6:
            steering_angle = 0.0
            linear_velocity = 0.0 
        else:
            target_orientation = np.arctan2(optimal_velocity_vector[1], optimal_velocity_vector[0])
            steering_angle = self._normalize_angle(target_orientation - robot_orientation)

        linear_velocity = np.clip(linear_velocity, 0.0, self.action_space.high[0]) 
        steering_angle = np.clip(steering_angle, self.action_space.low[1], self.action_space.high[1])

        return np.array([linear_velocity, steering_angle], dtype=np.float32)

    # --- NEW Helper: Update path progress and get current target waypoint ---
    def _update_path_progress_and_get_current_target(self, robot_x, robot_y):
        """
        Checks proximity to current target waypoint, advances if reached, and returns current target.
        Returns: (target_coords, status_message)
                 target_coords: tuple (x,y) or None if path ended/invalid
                 status_message: string describing outcome
        """
        if not self.current_planned_path: 
            return None, "Path Error (No Path)"

        if self.current_path_target_idx >= len(self.current_planned_path):
            return None, "Path Error (End of Path Reached Previously)"

        current_target_waypoint = np.array(self.current_planned_path[self.current_path_target_idx])
        robot_pos = np.array([robot_x, robot_y])
        dist_to_target = np.linalg.norm(robot_pos - current_target_waypoint)

        if dist_to_target < self.waypoint_achieved_threshold:
            self.current_path_target_idx += 1
            if self.current_path_target_idx >= len(self.current_planned_path):
                return None, "Path Error (Reached End of Path Now)" 

        if self.current_path_target_idx >= len(self.current_planned_path):
             return None, "Path Error (Path Ended)" # Should be caught by above

        final_target_waypoint_coords = tuple(self.current_planned_path[self.current_path_target_idx])
        return final_target_waypoint_coords, "Targeting Waypoint"


    # --- MODIFIED: _calculate_path_following_action using Bubble Rebound & Direct Waypoint Following ---
    def _calculate_path_following_action(self, robot_x, robot_y, robot_velocity_magnitude, robot_orientation, perceived_static_obstacles):
        """ Calculates the action based on direct RRT path following and perceived STATIC obstacles using Bubble Rebound"""
        current_target_point_coords, path_progress_status = self._update_path_progress_and_get_current_target(robot_x, robot_y)

        if current_target_point_coords is None:
            return np.array([0.0, 0.0]), path_progress_status 

        target_x, target_y = current_target_point_coords
        target_vector = np.array([target_x - robot_x, target_y - robot_y])
        target_distance = np.linalg.norm(target_vector)
        
        target_orientation_towards_waypoint = robot_orientation 
        if target_distance > 1e-6:
             target_orientation_towards_waypoint = np.arctan2(target_vector[1], target_vector[0])

        rebound_avoidance_orientation, is_rebound_active, min_distance_to_static_obs = self.bubble_rebound.compute_rebound_angle(
            robot_x, robot_y, robot_orientation, robot_velocity_magnitude, perceived_static_obstacles
        )

        final_target_orientation = target_orientation_towards_waypoint
        if is_rebound_active:
            final_target_orientation = rebound_avoidance_orientation
            # Optional SLERP for smoother transition to rebound_avoidance_orientation:
            # blend_factor = 1.0 # Or a value based on min_distance_to_static_obs
            # final_target_orientation = self._slerp_angle(
            # target_orientation_towards_waypoint, rebound_avoidance_orientation, blend_factor
            # )

        steering_angle = self._normalize_angle(final_target_orientation - robot_orientation)
        steering_angle = np.clip(steering_angle, self.action_space.low[1], self.action_space.high[1])

        velocity = self.max_velocity
        max_steer = abs(self.action_space.high[1]) if self.action_space.high[1] > 1e-6 else np.pi
        norm_steer_mag = abs(steering_angle) / max_steer if max_steer > 1e-6 else 0.0
        steering_vel_factor = max(0.0, np.cos(norm_steer_mag * np.pi / 2)**0.5)

        proximity_vel_factor = 1.0
        if min_distance_to_static_obs < self.obstacle_slow_down_distance:
            inner_slowdown_dist = self.obstacle_slow_down_distance * 0.3
            if min_distance_to_static_obs < inner_slowdown_dist:
                 proximity_vel_factor = np.clip(min_distance_to_static_obs / inner_slowdown_dist, 0.0, 1.0)
            else:
                 slow_down_ratio = (min_distance_to_static_obs - inner_slowdown_dist) / (self.obstacle_slow_down_distance - inner_slowdown_dist)
                 proximity_vel_factor = 0.5 + 0.5 * slow_down_ratio 
                 proximity_vel_factor = np.clip(proximity_vel_factor, 0.2, 1.0)

        velocity = self.max_velocity * steering_vel_factor * proximity_vel_factor
        velocity = np.clip(velocity, self.min_velocity, self.max_velocity)
        action = np.array([velocity, steering_angle], dtype=np.float32)

        current_action_status = ""
        if is_rebound_active:
            current_action_status = "Avoiding static (Bubble Rebound)"
        elif np.linalg.norm(action) < 1e-5 and target_distance > self.waypoint_achieved_threshold : 
            current_action_status = "Following Path (Calculated Stop)"
        else:
            current_action_status = "Following Path (No Rebound)"

        return action, current_action_status


    def get_action(self, observation: dict):
        robot_x, robot_y, robot_velocity, robot_orientation, goal_x, goal_y = observation['base_observation']
        self.perceived_obstacles = self.obstacle_identifier.identify(observation)
        memory_updated = self._update_obstacle_memory(self.perceived_obstacles)

        perceived_static_obstacles = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.STATIC]
        perceived_dynamic_obstacles = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.DYNAMIC]

        current_robot_velocity_vector = robot_velocity * np.array([np.cos(robot_orientation), np.sin(robot_orientation)])
        current_robot_velocity_magnitude = robot_velocity

        distance_to_goal = np.linalg.norm(np.array([robot_x, robot_y]) - np.array([goal_x, goal_y]))
        if distance_to_goal < self.goal_threshold:
            self.status = "Goal Reached - Stopping"
            self.current_planned_path = None 
            self.current_path_target_idx = 0
            info = self._get_controller_info()
            info['goal_reached'] = True
            final_action = np.array([0.0, 0.0])
            if self.collecting_data:
                 self.collected_data.append((observation.copy(), final_action.copy()))
            return final_action, info

        needs_replan = False
        initial_status = "Checking Path..."
        if self.current_planned_path is None or len(self.current_planned_path) == 0: # Path can be empty
             initial_status = "No valid path"
             needs_replan = True
        elif self.current_path_target_idx >= len(self.current_planned_path):
             initial_status = "Reached end of path segments"
             if distance_to_goal > self.goal_threshold * 1.5: 
                 needs_replan = True
             else: 
                 self.status = "Near Goal (End of Path)" 
                 needs_replan = False 
        # elif self._is_path_invalidated(robot_x, robot_y, perceived_static_obstacles):
        elif self._is_path_invalidated(robot_x, robot_y, self.perceived_obstacles):
              initial_status = "Path invalidated by perceived obstacle"
              needs_replan = True

        if needs_replan:
            self.status = initial_status + " - Replanning..."
            # self.map_obstacles = list(self.discovered_static_obstacles)
            self.map_obstacles = list(self.discovered_static_obstacles + perceived_dynamic_obstacles)
            # self.path_planner.set_goal(goal_x, goal_y)
            new_path, nodes, parents = self.path_planner.plan_path(
                robot_x, robot_y, goal_x, goal_y, self.map_obstacles, smooth_path=True
            )
            if new_path and len(new_path) > 0: 
                 self.current_planned_path = new_path
                 self.current_rrt_nodes = nodes
                 self.current_rrt_parents = parents
                 self.current_path_target_idx = 0 
                 self.status = "Replanning Successful"
                #  print(f"Replanning Successful: {self.current_planned_path}")
            else:
                 self.status = "Replanning Failed"
                 self.current_planned_path = None
                 self.current_path_target_idx = 0
                 self.current_rrt_nodes = nodes
                 self.current_rrt_parents = parents

        path_available_for_following = self.current_planned_path is not None and \
                                       len(self.current_planned_path) > 0 and \
                                       self.current_path_target_idx < len(self.current_planned_path)

        action_generation_status = ""
        if not path_available_for_following:
            preferred_action = np.array([0.0, 0.0])
            if "Replanning Failed" in self.status:
                action_generation_status = self.status 
            elif self.current_planned_path and self.current_path_target_idx >= len(self.current_planned_path):
                 action_generation_status = "Path Complete - Stopping" 
            else: # Includes case where self.current_planned_path is None or empty
                action_generation_status = "No Path - Stopping"
            self.status = action_generation_status 
        else:
            preferred_action, action_generation_status = self._calculate_path_following_action(
                robot_x, robot_y,
                current_robot_velocity_magnitude,
                robot_orientation,
                perceived_static_obstacles
            )
            if "Path Error" in action_generation_status or "Reached End" in action_generation_status or "Path Complete" in action_generation_status:
                self.status = action_generation_status 
            else:
                self.status = action_generation_status 

        preferred_velocity_vector = self._action_to_velocity_vector(
            preferred_action[0], preferred_action[1], robot_orientation
        )

        robot_agent = OrcaAgent(
            position=np.array([robot_x, robot_y]),
            velocity=current_robot_velocity_vector,
            radius=self.robot_radius,
            max_speed=self.max_velocity,
            pref_velocity=preferred_velocity_vector
        )
        dynamic_orca_agents = []
        for obs in perceived_dynamic_obstacles:
            obs_pos = np.array(obs.get_position())
            direction_norm = np.linalg.norm(obs.direction)
            norm_direction = obs.direction / direction_norm if direction_norm > 1e-6 else np.array([0.0, 0.0])
            obs_velocity_vector = obs.velocity * norm_direction
            effective_radius = self._get_orca_effective_radius(obs.shape)
            collider_agent = OrcaAgent(
                position=obs_pos, velocity=obs_velocity_vector, radius=effective_radius,
                max_speed=obs.velocity * 1.5, pref_velocity=obs_velocity_vector
            )
            dynamic_orca_agents.append(collider_agent)

        t_horizon = self.orca_tau
        dt_step = self.env_dt if self.env_dt > 0 else 0.1

        optimal_velocity_vector = preferred_velocity_vector
        orca_status_suffix = ""

        if dynamic_orca_agents:
            try:
                orca_optimal_velocity, _ = calculate_orca_velocity(
                    agent=robot_agent, colliding_agents=dynamic_orca_agents, t=t_horizon, dt=dt_step
                )
                orca_speed = np.linalg.norm(orca_optimal_velocity)
                if orca_speed > self.max_velocity:
                     orca_optimal_velocity = (orca_optimal_velocity / orca_speed) * self.max_velocity

                if np.isnan(orca_optimal_velocity).any():
                     optimal_velocity_vector = preferred_velocity_vector
                     orca_status_suffix = " (ORCA NaN Fallback)"
                else:
                     optimal_velocity_vector = orca_optimal_velocity
                     orca_status_suffix = " (ORCA Active)"
            except InfeasibleError:
                optimal_velocity_vector = np.array([0.0, 0.0])
                orca_status_suffix = " (ORCA Infeasible - Stopping)"
            except Exception as e:
                optimal_velocity_vector = np.array([0.0, 0.0])
                orca_status_suffix = f" (ORCA Error - Stopping)"

            if self.collecting_data:
                 self.collected_data.append((observation.copy(), optimal_velocity_vector.copy()))

        if orca_status_suffix:
             if "Stopping" in orca_status_suffix or "Infeasible" in orca_status_suffix or "Error" in orca_status_suffix :
                  self.status = orca_status_suffix.strip() 
             elif not ( "Error" in self.status or "Stopping" in self.status or "Failed" in self.status or "Complete" in self.status or "End of Path" in self.status ):
                    self.status += orca_status_suffix 
        elif not ("Error" in self.status or "Stopping" in self.status or "Failed" in self.status or "Complete" in self.status or "End of Path" in self.status):
            if "Bubble Rebound" in self.status:
                pass 
            elif "Following Path" in self.status : 
                 self.status = "Following Path (No dynamic obstacles)"
            # Ensure a sensible status if it was e.g. "Replanning Successful" but now no path/action
            elif not path_available_for_following and np.linalg.norm(preferred_action) < 1e-6 :
                if "Replanning Failed" not in self.status and "No Path" not in self.status: # if not already set to a stop state
                    self.status = "No Action / Path Ended"


        final_action = self._velocity_vector_to_action(optimal_velocity_vector, robot_orientation)

        controller_info = self._get_controller_info()
        controller_info["status"] = self.status
        controller_info["discovered_static_obstacles"] = self.discovered_static_obstacles
        controller_info["map_obstacles_used_for_plan"] = self.map_obstacles

        return final_action, controller_info

    def _is_path_invalidated(self, robot_x, robot_y, obstacles_to_check: list):
        if not self.current_planned_path or len(self.current_planned_path) < 2:
             return False 
        robot_pos = np.array([robot_x, robot_y])
        min_dist_sq = float('inf')
        closest_segment_idx = 0
        
        temp_search_start = 0
        temp_search_end = len(self.current_planned_path) - 2 
        
        if temp_search_end < temp_search_start : 
            return False

        for i in range(temp_search_start, temp_search_end + 1):
             p1 = np.array(self.current_planned_path[i])
             p2 = np.array(self.current_planned_path[i+1])
             seg_vec = p2 - p1
             seg_len_sq = np.dot(seg_vec, seg_vec)
             if seg_len_sq < 1e-9: 
                  dist_sq = np.sum((robot_pos - p1)**2)
             else:
                  t = np.clip(np.dot(robot_pos - p1, seg_vec) / seg_len_sq, 0, 1)
                  closest_point_on_segment = p1 + t * seg_vec
                  dist_sq = np.sum((robot_pos - closest_point_on_segment)**2)
             if dist_sq < min_dist_sq:
                  min_dist_sq = dist_sq
                  closest_segment_idx = i 

        start_check_idx = closest_segment_idx
        end_check_idx = min(len(self.current_planned_path) - 2, start_check_idx + self.path_invalidation_check_horizon)
        
        if start_check_idx < 0 or start_check_idx >= len(self.current_planned_path) -1 : 
            return False

        for i in range(start_check_idx, end_check_idx + 1):
             p1_tuple = tuple(self.current_planned_path[i])
             p2_tuple = tuple(self.current_planned_path[i+1])
             for obs in obstacles_to_check:
                  obs_x, obs_y = obs.get_position()
                  if obs.shape.intersects_segment(p1_tuple, p2_tuple, self.robot_radius, obs_x, obs_y):
                       return True
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
        return {
            "status": self.status,
            "planned_path": self.current_planned_path.copy() if self.current_planned_path else None,
            "rrt_nodes": self.current_rrt_nodes,
            "rrt_parents": self.current_rrt_parents,
            "target_idx": self.current_path_target_idx,
            "discovered_static_obstacles": [obs for obs in self.discovered_static_obstacles],
            "map_obstacles_used_for_plan": [obs for obs in self.map_obstacles],
            }

# --- END OF FILE indoor_robot_controller.py ---