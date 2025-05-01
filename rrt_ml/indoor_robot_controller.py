# indoor_robot_controller.py

import time
import math
import numpy as np
from indoor_robot_env import IndoorRobotEnv
from components.obstacle import Obstacle, ObstacleType
from components.shape import Circle, Rectangle, Shape
from utils.ray_tracing_algorithm import RayTracingAlgorithm
from utils.rrt_planner import RRTPathPlanner
# Remove or comment out WaitingRule if ORCA completely replaces its action logic
# from utils.waiting_rule import WaitingRule
from utils.obstacle_identifier import ObstacleIdentifier
import pickle # Import pickle for saving data

# -- Use ML model --
from utils.ml_avoidance import SimpleMLPAvoidance

# ... rest of the imports ...

class IndoorRobotController:
    def __init__(self, env: IndoorRobotEnv): # Type hint env
        # ... existing initializations ...
        self.width = env.width
        self.height = env.height
        self.robot_radius = env.robot_radius
        self.action_space = env.action_space
        self.observation_space = env.observation_space # Pass to identifier
        self.env_dt = env.dt # Store environment timestep if available

        # Initialize components
        self.obstacle_identifier = ObstacleIdentifier(self.observation_space, env.max_obstacles_in_observation)
        self.ray_tracer = RayTracingAlgorithm(self.width, self.height, self.robot_radius)
        # self.waiting_rule = WaitingRule(self.robot_radius) # Keep if needed for detection, disable if ORCA handles all
        self.path_planner = RRTPathPlanner(self.width, self.height, self.robot_radius)
        self.ml_avoidance = SimpleMLPAvoidance(obs_robot_state_size=env.OBS_ROBOT_STATE_SIZE, obs_obstacle_data_size=env.OBS_OBSTACLE_DATA_SIZE) # Initialize ML model

        # Controller parameters (mostly same)
        self.goal_threshold = self.robot_radius + 5
        self.max_velocity = self.action_space.high[0]
        self.min_velocity = 0.3 # Keep minimum linear velocity? ORCA might output zero.
        self.obstacle_slow_down_distance = self.robot_radius * 5
        self.obstacle_avoid_distance = self.robot_radius * 3 # Used for static avoidance blending
        self.lookahead_distance = self.robot_radius * 4
        self.path_invalidation_check_horizon = 5

        # Controller state
        self.current_planned_path = None
        self.current_rrt_nodes = None
        self.current_rrt_parents = None
        self.current_path_target_idx = 0
        self.perceived_obstacles = [] # List of Obstacle objects
        self.map_obstacles = [] # List of Obstacle objects used for last plan
        self.is_waiting = False # May not be needed if ORCA handles waits implicitly
        self.last_replanning_time = -np.inf
        self.status = "Initializing"
        # -----------------------
    

    def reset(self):
         # ... existing reset logic ...
        #  self.last_robot_pos = None
        #  self.last_time = None
        #  self.current_robot_velocity_vector = np.array([0.0, 0.0])
         self.current_planned_path = None
         self.current_rrt_nodes = None
         self.current_rrt_parents = None
         self.current_path_target_idx = 0
         self.perceived_obstacles = []
         self.map_obstacles = []
         self.is_waiting = False
         self.last_replanning_time = -np.inf
         self.status = "Reset"
         # Do NOT reset collected_data here

    # ... (Data collection methods remain the same) ...

    # --- ( _calculate_path_following_action remains largely the same, handles STATIC obstacles ) ---
    def _calculate_path_following_action(self, robot_x, robot_y, robot_orientation, perceived_obstacles):
        # ... (Keep the existing logic for path following and STATIC avoidance) ...
        # ... (This function calculates the action the robot *would* take ignoring dynamic obstacles) ...
        # Returns: action = np.array([velocity, steering_angle]), calculation_status

        # --- Start copied logic ---
        calculation_status = "Following Path (Calculating)"
        if not self.current_planned_path: # Check if path exists before getting lookahead
            calculation_status = "Path Error (No Path) - Stopping"
            return np.array([0.0, 0.0]), calculation_status


        lookahead_point, self.current_path_target_idx = self._get_lookahead_point(robot_x, robot_y)

        if lookahead_point is None:
            calculation_status = "Path Error (Lookahead) - Stopping"
            # print(calculation_status)
            # self.current_planned_path = None # Don't modify global state here
            return np.array([0.0, 0.0]), calculation_status

        target_x, target_y = lookahead_point
        target_vector = np.array([target_x - robot_x, target_y - robot_y])
        target_distance = np.linalg.norm(target_vector)
        target_orientation = robot_orientation # Default if distance is zero

        if target_distance > 1e-6:
             target_orientation = np.arctan2(target_vector[1], target_vector[0])
        # If target_distance is near zero, check next point if available
        elif self.current_planned_path and self.current_path_target_idx < len(self.current_planned_path) - 1:
             next_target = self.current_planned_path[self.current_path_target_idx + 1]
             target_vector = np.array([next_target[0] - robot_x, next_target[1] - robot_y])
             if np.linalg.norm(target_vector) > 1e-6:
                  target_orientation = np.arctan2(target_vector[1], target_vector[0])


        # --- Reactive Static Obstacle Avoidance (Step 5 adapted) ---
        final_target_orientation = target_orientation
        min_dist_to_static_boundary = float('inf')
        avoiding_obstacle = None
        current_min_avoid_eff_dist = float('inf') # Track minimum eff_dist for *chosen* avoidance candidate

        # Filter for STATIC obstacles only here
        perceived_static_obstacles = [obs for obs in perceived_obstacles if obs.type == ObstacleType.STATIC]
        for obstacle in perceived_static_obstacles:
            # Ensure get_position returns numpy array
            obs_pos = np.array(obstacle.get_position())
            robot_pos = np.array([robot_x, robot_y])

            eff_dist = obstacle.shape.get_efficient_distance(robot_x, robot_y, obs_pos[0], obs_pos[1]) - self.robot_radius
            min_dist_to_static_boundary = min(min_dist_to_static_boundary, eff_dist)

            if eff_dist < self.obstacle_avoid_distance:
                vec_to_obs = obs_pos - robot_pos
                # Check if obstacle is roughly in front based on target direction
                angle_to_obs = np.arctan2(vec_to_obs[1], vec_to_obs[0])
                angle_diff_to_target = abs(self._normalize_angle(angle_to_obs - target_orientation))

                # Increased avoidance cone slightly
                if angle_diff_to_target < np.pi / 1.8: # Wider cone?
                    # Choose the closest one within the cone
                    if avoiding_obstacle is None or eff_dist < current_min_avoid_eff_dist:
                         avoiding_obstacle = obstacle
                         current_min_avoid_eff_dist = eff_dist # Update min distance for chosen one

        if avoiding_obstacle is not None:
            # Use the final eff_dist of the *chosen* obstacle for blending
            final_avoid_eff_dist = current_min_avoid_eff_dist
            calculation_status = f"Avoiding static {type(avoiding_obstacle.shape).__name__} (Calculating)"
            # Ensure goal is available for avoid_static_obstacle
            goal_pos = self.path_planner.goal if self.path_planner.goal else np.array([robot_x, robot_y]) # Use current pos if goal unknown
            avoidance_orientation = self.ray_tracer.avoid_static_obstacle(
                robot_x, robot_y, robot_orientation, avoiding_obstacle, goal_pos[0], goal_pos[1] # Pass goal explicitly
            )
            blend_factor = 1.0 - np.clip(final_avoid_eff_dist / self.obstacle_avoid_distance, 0.0, 1.0)
            blend_factor = blend_factor**2
            final_target_orientation = self._slerp_angle(target_orientation, avoidance_orientation, blend_factor)

        # --- Calculate Control Action (Step 6 adapted) ---
        steering_angle = self._normalize_angle(final_target_orientation - robot_orientation)
        steering_angle = np.clip(steering_angle, self.action_space.low[1], self.action_space.high[1])

        velocity = self.max_velocity
        max_steer = abs(self.action_space.high[1])
        norm_steer_mag = abs(steering_angle) / max_steer if max_steer > 1e-6 else 0
        # Gentler steering slowdown factor
        steering_vel_factor = max(0.2, np.cos(norm_steer_mag * np.pi / 2)**0.5) # Adjusted factor

        # Velocity scaling based on proximity to *any* perceived obstacle (STATIC only for this calculation)
        # min_dist_overall_eff = min_dist_to_static_boundary # Start with closest static

        # Apply velocity scaling based on proximity to STATIC obstacles
        if min_dist_to_static_boundary < self.obstacle_avoid_distance:
             # Smoother proximity factor
             proximity_vel_factor = np.clip(min_dist_to_static_boundary / self.obstacle_avoid_distance, 0.1, 1.0)**0.5
        elif min_dist_to_static_boundary < self.obstacle_slow_down_distance:
            slow_down_ratio = (min_dist_to_static_boundary - self.obstacle_avoid_distance) / (self.obstacle_slow_down_distance - self.obstacle_avoid_distance)
            proximity_vel_factor = 0.6 + 0.4 * slow_down_ratio # Start slowdown earlier, less aggressive
            proximity_vel_factor = np.clip(proximity_vel_factor, 0.6, 1.0)
        else:
            proximity_vel_factor = 1.0

        velocity = self.max_velocity * steering_vel_factor * proximity_vel_factor
        velocity = np.clip(velocity, self.min_velocity, self.max_velocity) # Ensure min velocity if desired

        action = np.array([velocity, steering_angle], dtype=np.float32)

        # Update status if not already set to avoiding/error
        if "Calculating" in calculation_status:
             # Check if path following is active (i.e., not stopped due to error)
             if np.linalg.norm(action) > 1e-6 or self.current_planned_path:
                 calculation_status = "Following Path (Static Avoidance)"
             else:
                 calculation_status = "Path Error (No Action)" # Or similar error status


        return action, calculation_status
        # --- End copied logic ---


    def get_action(self, observation: np.ndarray):
        # current_time = time.time()
        # Initialize last_time on first call
        # if self.last_time is None:
        #      self.last_time = current_time

        # --- 1. Perceive Environment & Estimate Velocity ---
        robot_x, robot_y, robot_orientation, goal_x, goal_y = observation[:IndoorRobotEnv.OBS_ROBOT_STATE_SIZE]
        self.perceived_obstacles = self.obstacle_identifier.identify(observation)


        # --- 2. Check Goal Reached ---
        distance_to_goal = np.linalg.norm(np.array([robot_x, robot_y]) - np.array([goal_x, goal_y]))
        if distance_to_goal < self.goal_threshold:
            self.status = "Goal Reached - Stopping"
            self.current_planned_path = None
            self.current_path_target_idx = 0
            # Reset velocity estimation state upon reaching goal? Optional.
            # self.last_robot_pos = None
            # self.last_time = None
            info = self._get_controller_info()
            info['goal_reached'] = True
            final_action = np.array([0.0, 0.0]) # Ensure stop action
            return final_action, info

        # --- 3. Replanning Logic ---
        # ... (Keep existing replanning logic) ...
        needs_replan = False
        initial_status = "Checking Path..."
        if self.current_planned_path is None or len(self.current_planned_path) <= 1:
             initial_status = "No valid path"
             needs_replan = True
        elif self.current_path_target_idx >= len(self.current_planned_path):
             # This case should ideally be handled by goal check or lookahead logic, but good fallback
             initial_status = "Reached end of path"
             needs_replan = True # Replan to confirm if goal is reached or if stuck
             # Maybe better: check distance to goal here again?
             if distance_to_goal > self.goal_threshold * 1.5: # If far from goal but end of path
                 needs_replan = True
             else: # Likely near goal, let goal check handle it
                 needs_replan = False # Avoid replanning loop near goal
                 self.status = "Near Goal (End of Path)"

        elif self._is_path_invalidated(robot_x, robot_y):
             initial_status = "Path invalidated by new obstacle"
             needs_replan = True

        if needs_replan:
            self.status = initial_status + " - Replanning..."
            # print(self.status) # Optional print
            current_static_map = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.STATIC]
            self.map_obstacles = current_static_map # Update map obstacles used for planning

            self.path_planner.set_goal(goal_x, goal_y) # Ensure goal is set

            new_path, nodes, parents = self.path_planner.plan_path(robot_x, robot_y, goal_x, goal_y, self.map_obstacles)

            if new_path and len(new_path) > 1:
                 self.current_planned_path = new_path
                 self.current_rrt_nodes = nodes
                 self.current_rrt_parents = parents
                 self.current_path_target_idx = 0 # Reset index for new path
                #  self.last_replanning_time = current_time
                 self.status = "Replanning Successful"
            else:
                 self.status = "Replanning Failed"
                 # print(self.status) # Optional print
                 self.current_planned_path = None # Clear failed path
                 self.current_path_target_idx = 0
                 # Keep RRT tree for visualization even on failure
                 self.current_rrt_nodes = nodes
                 self.current_rrt_parents = parents
                 # Decision to stop is made later based on path availability

        # Check path availability *after* potential replanning
        path_available = self.current_planned_path is not None and len(self.current_planned_path) > 1

        # --- 4. Calculate Preferred Velocity (based on path following & static avoidance) ---
        if not path_available:
            # If no path after replanning, preferred action is to stop
            preferred_action = np.array([0.0, 0.0])
            if self.status != "Replanning Failed": self.status = "No Path - Stopping"
        else:
            # Calculate the action considering only path following and static obstacles
            preferred_action, self.status = self._calculate_path_following_action(
                robot_x, robot_y, robot_orientation, self.perceived_obstacles
            )


        perceived_dynamic_obstacles = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.DYNAMIC]
        if perceived_dynamic_obstacles:
            # --- 5. Dynamic Obstacle Avoidance ---
            # Use ML model for dynamic obstacle avoidance
            ml_action = self.ml_avoidance.predict(
                observation
            )
            # Combine with preferred action (path following)
            final_action = np.clip(ml_action, self.action_space.low, self.action_space.high)
            self.status = "Dynamic Avoidance (ML)"
        else:
            final_action = preferred_action # No dynamic obstacles, use preferred action
        

        # --- 8. Return Action and Info ---
        controller_info = self._get_controller_info()
        # Update status in info dict if it changed
        controller_info["status"] = self.status
        # Add estimated velocity and preferred velocity for debugging?
        # controller_info["estimated_velocity"] = current_robot_velocity_vector
        # controller_info["preferred_velocity"] = preferred_velocity_vector
        # controller_info["optimal_velocity"] = optimal_velocity_vector


        return final_action, controller_info


    # --- Helper functions (_get_lookahead_point, _is_path_invalidated, _normalize_angle, _slerp_angle, _get_controller_info) ---
    # Paste the existing helper functions here (make sure they are indented correctly within the class)
    def _get_lookahead_point(self, robot_x, robot_y):
        # This logic remains the same as it works on the path (list of points)
        if not self.current_planned_path or len(self.current_planned_path) < 1: # Need at least one point
            # print("Lookahead: No path")
            return None, self.current_path_target_idx

        robot_pos = np.array([robot_x, robot_y])
        current_target_idx = self.current_path_target_idx

        # Ensure target index is valid
        current_target_idx = min(max(0, current_target_idx), len(self.current_planned_path) - 1)


        # --- Update Path Target Index ---
        # Find the segment on the path closest to the robot
        search_start_idx = max(0, current_target_idx - 2) # Look back a bit more
        search_end_idx = min(len(self.current_planned_path) - 2, current_target_idx + 5) # Look ahead a bit
        min_dist_to_segment_sq = float('inf')
        closest_segment_idx = current_target_idx # Default
        projection_t = 0.0 # Projection factor onto the closest segment

        if len(self.current_planned_path) < 2: # Path is just one point
             # If only one point, target is that point, check distance
             dist_to_target_sq = np.sum((robot_pos - np.array(self.current_planned_path[0]))**2)
             if dist_to_target_sq < (self.lookahead_distance * 0.5)**2: # If close enough, consider it the target
                 current_target_idx = 0
             # No lookahead point possible beyond the single point
             return self.current_planned_path[0], 0


        for i in range(search_start_idx, search_end_idx + 1):
            # Ensure indices are valid before accessing
            if i >= len(self.current_planned_path) - 1:
                break # Stop if index goes out of bounds

            p1 = np.array(self.current_planned_path[i])
            p2 = np.array(self.current_planned_path[i+1])
            seg_vec = p2 - p1
            seg_len_sq = np.dot(seg_vec, seg_vec)

            if seg_len_sq < 1e-12: # Segment is effectively a point
                 dist_sq = np.sum((robot_pos - p1)**2)
                 t = 0.0
            else:
                 # Project robot position onto the line defined by the segment
                 t = np.dot(robot_pos - p1, seg_vec) / seg_len_sq
                 # Find the closest point on the *finite segment*
                 t_clamped = np.clip(t, 0, 1)
                 closest_point_on_segment = p1 + t_clamped * seg_vec
                 dist_sq = np.sum((robot_pos - closest_point_on_segment)**2)

            if dist_sq < min_dist_to_segment_sq:
                 min_dist_to_segment_sq = dist_sq
                 closest_segment_idx = i
                 projection_t = t # Store the unclamped projection factor

        # --- Advance Target Index ---
        # Heuristic: If robot is past the midpoint (t > 0.5) of the segment *before* the current target,
        # or significantly past the start of the current target segment, advance the target.
        if closest_segment_idx == current_target_idx and projection_t > 0.1:
             # If projected onto current target segment and moved along it
             if current_target_idx < len(self.current_planned_path) - 1:
                  current_target_idx += 1
        elif closest_segment_idx > current_target_idx:
             # If closest segment is already ahead of the target, update target
              current_target_idx = min(closest_segment_idx + 1, len(self.current_planned_path) - 1)
        elif closest_segment_idx == current_target_idx - 1 and projection_t > 0.5 : # Check projection on segment *before* target
             # If significantly past the midpoint of the previous segment
             if current_target_idx < len(self.current_planned_path) -1:
                 current_target_idx += 1


        # Ensure target index stays valid
        current_target_idx = min(max(0, current_target_idx), len(self.current_planned_path) - 1)
        self.current_path_target_idx = current_target_idx # Update controller state


        # --- Find Lookahead Point ---
        # Start searching for the lookahead point from the segment associated with the (potentially updated) target index, or slightly before.
        search_start_node_idx = max(0, current_target_idx -1) # Start from segment beginning at this node index

        # Find the closest point on the path (on the relevant segment) to start measuring lookahead distance from
        start_measure_point = robot_pos # Default to robot pos if segment finding fails
        if len(self.current_planned_path) >= 2:
             p1_search = np.array(self.current_planned_path[search_start_node_idx])
             # Ensure index i+1 is valid
             p2_search_idx = min(search_start_node_idx + 1, len(self.current_planned_path) - 1)
             p2_search = np.array(self.current_planned_path[p2_search_idx])

             seg_vec_search = p2_search - p1_search
             seg_len_sq_search = np.dot(seg_vec_search, seg_vec_search)

             if seg_len_sq_search < 1e-12:
                  start_measure_point = p1_search
             else:
                  t_search = np.dot(robot_pos - p1_search, seg_vec_search) / seg_len_sq_search
                  t_clamped_search = np.clip(t_search, 0, 1)
                  start_measure_point = p1_search + t_clamped_search * seg_vec_search


        # Iterate forward along the path from search_start_node_idx
        cumulative_dist = 0.0
        lookahead_point_found = None

        # Distance from the projected start point to the end of the starting segment
        dist_on_first_segment = np.linalg.norm(np.array(self.current_planned_path[min(search_start_node_idx + 1, len(self.current_planned_path)-1)]) - start_measure_point)

        if dist_on_first_segment >= self.lookahead_distance:
            # Lookahead point is on the first segment itself
            p1 = start_measure_point # Start from the projection
            p2 = np.array(self.current_planned_path[min(search_start_node_idx + 1, len(self.current_planned_path)-1)])
            vec = p2 - p1
            norm_vec = vec / dist_on_first_segment if dist_on_first_segment > 1e-6 else vec
            lookahead_point_found = tuple(p1 + norm_vec * self.lookahead_distance)

        else:
            # Need to check subsequent segments
            cumulative_dist += dist_on_first_segment
            for i in range(search_start_node_idx + 1, len(self.current_planned_path) - 1):
                p1 = np.array(self.current_planned_path[i])
                p2 = np.array(self.current_planned_path[i+1])
                segment_vec = p2 - p1
                segment_len = np.linalg.norm(segment_vec)

                if cumulative_dist + segment_len >= self.lookahead_distance:
                    # Lookahead point lies on this segment (p1 to p2)
                    remaining_dist = self.lookahead_distance - cumulative_dist
                    ratio = remaining_dist / segment_len if segment_len > 1e-6 else 0
                    lookahead_point_found = tuple(p1 + ratio * segment_vec)
                    break
                else:
                    cumulative_dist += segment_len

        # If loop finishes and no lookahead point found (e.g., path shorter than lookahead distance)
        if lookahead_point_found is None:
            lookahead_point_found = tuple(self.current_planned_path[-1]) # Target the end of the path


        # print(f"Lookahead: Idx={self.current_path_target_idx}, Point={lookahead_point_found}")
        return lookahead_point_found, self.current_path_target_idx


    def _is_path_invalidated(self, robot_x, robot_y):
        if not self.current_planned_path or len(self.current_planned_path) < 2:
             return False # No path or path too short to check

        robot_pos = np.array([robot_x, robot_y])

        # --- Find segment closest to robot (more robustly) ---
        min_dist_sq = float('inf')
        closest_segment_idx = 0 # Default to first segment
        # Check segments around the current target index first
        search_center = self.current_path_target_idx
        search_radius = 5 # How many segments around the target to check
        search_start = max(0, search_center - search_radius)
        search_end = min(len(self.current_planned_path) - 2, search_center + search_radius)

        # Ensure valid range if path is short
        search_end = max(search_start, search_end)


        for i in range(search_start, search_end + 1):
             # Check index bounds
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
        # --- End find closest segment ---

        # Check segments ahead from the closest segment found
        start_check_idx = closest_segment_idx
        # Check a fixed number of segments or until the end of the path
        end_check_idx = min(len(self.current_planned_path) - 2, start_check_idx + self.path_invalidation_check_horizon)

        # Ensure start index is valid
        start_check_idx = min(start_check_idx, len(self.current_planned_path) - 2)
        if start_check_idx < 0: return False # Cannot check if path only has 1 point


        for i in range(start_check_idx, end_check_idx + 1):
             # Ensure indices are valid
             if i >= len(self.current_planned_path) - 1: break

             p1_tuple = tuple(self.current_planned_path[i])
             p2_tuple = tuple(self.current_planned_path[i+1])

             # Check this path segment against *currently perceived obstacles* (static and dynamic)
             for obs in self.perceived_obstacles:
                  # Use the obstacle's intersects_segment method
                  # The RRT planner likely checked against map_obstacles (mostly static)
                  # Here we check against real-time perceived obstacles
                  if obs.intersects_segment(p1_tuple, p2_tuple, self.robot_radius):
                       # print(f"Path invalidated: Segment {i} ({p1_tuple} -> {p2_tuple}) blocked by perceived {type(obs.shape).__name__} near ({obs.x:.1f},{obs.y:.1f})")
                       return True # Path is blocked ahead

        return False


    def _normalize_angle(self, angle):
        while angle > np.pi: angle -= 2 * np.pi
        while angle < -np.pi: angle += 2 * np.pi
        return angle

    def _slerp_angle(self, a1, a2, t):
        # Ensure angles are in [-pi, pi]
        a1 = self._normalize_angle(a1)
        a2 = self._normalize_angle(a2)
        diff = self._normalize_angle(a2 - a1)
        # Linear interpolation on the normalized difference is usually fine for angles
        interpolated = self._normalize_angle(a1 + diff * t)
        return interpolated


    def _get_controller_info(self):
        return {
            "status": self.status,
            "planned_path": self.current_planned_path,
            "rrt_nodes": self.current_rrt_nodes,
            "rrt_parents": self.current_rrt_parents,
            "target_idx": self.current_path_target_idx,
            # "is_waiting": self.is_waiting # Add waiting status for debugging/info
            # "rays": self.last_ray_viz_points (if rays implemented fully)
            }