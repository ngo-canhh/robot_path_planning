import time
import math
import numpy as np
from indoor_robot_env import IndoorRobotEnv
from components.obstacle import Obstacle, ObstacleType
from components.shape import Circle, Rectangle
from utils.ray_tracing_algorithm import RayTracingAlgorithm
from utils.rrt_planner import RRTPathPlanner
from utils.waiting_rule import WaitingRule
from utils.obstacle_identifier import ObstacleIdentifier
import pickle # Import pickle for saving data

# -- Use ML model --
from utils.ml_avoidance import SimpleMLPAvoidance

# --- IndoorRobotController ---

class IndoorRobotController:
    def __init__(self, env: IndoorRobotEnv): # Type hint env
        self.width = env.width
        self.height = env.height
        self.robot_radius = env.robot_radius
        self.action_space = env.action_space
        self.observation_space = env.observation_space # Pass to identifier

        # Initialize components
        self.obstacle_identifier = ObstacleIdentifier(self.observation_space, env.max_obstacles_in_observation)
        self.ray_tracer = RayTracingAlgorithm(self.width, self.height, self.robot_radius) # Ray tracer less useful without proper intersection logic
        self.waiting_rule = WaitingRule(self.robot_radius)
        self.path_planner = RRTPathPlanner(self.width, self.height, self.robot_radius)
        # -- Use ML model --
        self.ml_avoidance = SimpleMLPAvoidance(obs_robot_state_size=5, obs_obstacle_data_size=9)

        # Controller parameters (mostly same)
        self.goal_threshold = self.robot_radius + 5
        self.max_velocity = self.action_space.high[0]
        self.min_velocity = 0.3
        self.obstacle_slow_down_distance = self.robot_radius * 5
        self.obstacle_avoid_distance = self.robot_radius * 3
        self.lookahead_distance = self.robot_radius * 4
        self.path_invalidation_check_horizon = 5

        # Controller state
        self.current_planned_path = None
        self.current_rrt_nodes = None
        self.current_rrt_parents = None
        self.current_path_target_idx = 0
        self.perceived_obstacles = [] # List of Obstacle objects
        self.map_obstacles = [] # List of Obstacle objects used for last plan
        self.is_waiting = False
        self.last_replanning_time = -np.inf
        self.status = "Initializing"

        # --- Data Collection ---
        self.collecting_data = False # Flag to enable/disable collection
        self.collected_data = [] # List to store (state, action) pairs
        # -----------------------

    def reset(self):
         self.current_planned_path = None
         self.current_rrt_nodes = None
         self.current_rrt_parents = None
         self.current_path_target_idx = 0
         self.perceived_obstacles = []
         self.map_obstacles = []
         self.is_waiting = False
         self.last_replanning_time = -np.inf
         self.status = "Reset"
         # Do NOT reset collected_data here if you want to collect across episodes
         # Reset it before starting a new data collection campaign

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


    # --- Refactored Action Calculation (Steps 5 & 6) ---
    def _calculate_path_following_action(self, robot_x, robot_y, robot_orientation, perceived_obstacles):
        """
        Calculates the desired action based on path following and static avoidance.
        This is the logic originally in steps 5 and 6 of get_action.
        Returns: (action, status_string)
        """
        calculation_status = "Following Path (Calculating)"

        # --- Path Following (Step 5 adapted) ---
        lookahead_point, self.current_path_target_idx = self._get_lookahead_point(robot_x, robot_y)

        if lookahead_point is None:
            calculation_status = "Path Error (Lookahead) - Stopping"
            # print(calculation_status)
            # self.current_planned_path = None # Don't modify global state here
            return np.array([0.0, 0.0]), calculation_status

        target_x, target_y = lookahead_point
        target_vector = np.array([target_x - robot_x, target_y - robot_y])
        target_distance = np.linalg.norm(target_vector)

        if target_distance < 1e-6:
            if self.current_planned_path and self.current_path_target_idx < len(self.current_planned_path) - 1:
                next_target = self.current_planned_path[self.current_path_target_idx + 1]
                target_vector = np.array([next_target[0] - robot_x, next_target[1] - robot_y])
                target_orientation = np.arctan2(target_vector[1], target_vector[0]) if np.linalg.norm(target_vector) > 1e-6 else robot_orientation
            else:
                target_orientation = robot_orientation
        else:
            target_orientation = np.arctan2(target_vector[1], target_vector[0])

        # --- Reactive Static Obstacle Avoidance (Step 5 adapted) ---
        final_target_orientation = target_orientation
        min_dist_to_static_boundary = float('inf')
        avoiding_obstacle = None
        current_min_avoid_eff_dist = float('inf') # Track minimum eff_dist for *chosen* avoidance candidate

        perceived_static_obstacles = [obs for obs in perceived_obstacles if obs.type == ObstacleType.STATIC]
        for obstacle in perceived_static_obstacles:
            eff_dist = obstacle.shape.get_efficient_distance(robot_x, robot_y, *obstacle.get_position()) - self.robot_radius
            min_dist_to_static_boundary = min(min_dist_to_static_boundary, eff_dist)

            if eff_dist < self.obstacle_avoid_distance:
                vec_to_obs = obstacle.get_position() - np.array([robot_x, robot_y])
                # Check if obstacle is roughly in front based on target direction
                angle_to_obs = np.arctan2(vec_to_obs[1], vec_to_obs[0])
                angle_diff_to_target = abs(self._normalize_angle(angle_to_obs - target_orientation))

                if angle_diff_to_target < np.pi / 1.9: # Avoidance cone
                    # Choose the closest one within the cone
                    if avoiding_obstacle is None or eff_dist < current_min_avoid_eff_dist:
                         avoiding_obstacle = obstacle
                         current_min_avoid_eff_dist = eff_dist # Update min distance for chosen one

        if avoiding_obstacle is not None:
            # Use the final eff_dist of the *chosen* obstacle for blending
            final_avoid_eff_dist = current_min_avoid_eff_dist
            calculation_status = f"Avoiding static {type(avoiding_obstacle.shape).__name__} (Calculating)"
            avoidance_orientation = self.ray_tracer.avoid_static_obstacle(
                robot_x, robot_y, robot_orientation, avoiding_obstacle, self.path_planner.goal[0], self.path_planner.goal[1] # Pass goal explicitly
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
        steering_vel_factor = max(0.1, np.cos(norm_steer_mag * np.pi / 2))

        # Velocity scaling based on proximity to *any* perceived obstacle
        min_dist_overall_eff = min_dist_to_static_boundary # Start with closest static
        perceived_dynamic_obstacles = [obs for obs in perceived_obstacles if obs.type == ObstacleType.DYNAMIC]
        for obs in perceived_dynamic_obstacles:
             eff_dist = obs.shape.get_efficient_distance(robot_x, robot_y, *obs.get_position()) - self.robot_radius
             min_dist_overall_eff = min(min_dist_overall_eff, eff_dist)

        # Apply velocity scaling based on proximity
        if min_dist_overall_eff < self.obstacle_avoid_distance:
            proximity_vel_factor = np.clip(min_dist_overall_eff / self.obstacle_avoid_distance, 0.0, 1.0)**0.75
        elif min_dist_overall_eff < self.obstacle_slow_down_distance:
            slow_down_ratio = (min_dist_overall_eff - self.obstacle_avoid_distance) / (self.obstacle_slow_down_distance - self.obstacle_avoid_distance)
            proximity_vel_factor = 0.5 + 0.5 * slow_down_ratio
            proximity_vel_factor = np.clip(proximity_vel_factor, 0.5, 1.0)
        else:
            proximity_vel_factor = 1.0

        velocity = self.max_velocity * steering_vel_factor * proximity_vel_factor
        velocity = np.clip(velocity, self.min_velocity, self.max_velocity)

        action = np.array([velocity, steering_angle], dtype=np.float32)

        # Update status if not already set to avoiding/error
        if "Calculating" in calculation_status:
             calculation_status = "Following Path"

        return action, calculation_status
    # ----------------------------------------------------


    def get_action(self, observation: np.ndarray):
        current_time = time.time()
        # self.status = "Processing" # Status will be set by calculations below

        # 1. Perceive Environment
        robot_x, robot_y, robot_orientation, goal_x, goal_y = observation[:IndoorRobotEnv.OBS_ROBOT_STATE_SIZE]
        self.perceived_obstacles = self.obstacle_identifier.identify(observation)

        # 2. Check Goal Reached
        distance_to_goal = np.linalg.norm(np.array([robot_x, robot_y]) - np.array([goal_x, goal_y]))
        if distance_to_goal < self.goal_threshold:
            self.status = "Goal Reached - Stopping"
            self.current_planned_path = None
            self.current_path_target_idx = 0
            info = self._get_controller_info()
            info['goal_reached'] = True
            return np.array([0.0, 0.0]), info

        # 3. Replanning Logic
        needs_replan = False
        initial_status = "Checking Path..."
        if self.current_planned_path is None or len(self.current_planned_path) <= 1:
             initial_status = "No valid path"
             needs_replan = True
        elif self.current_path_target_idx >= len(self.current_planned_path):
             initial_status = "Target index out of bounds"
             needs_replan = True
             self.current_planned_path = None
        elif self._is_path_invalidated(robot_x, robot_y):
             initial_status = "Path invalidated by new obstacle"
             needs_replan = True

        if needs_replan:
            self.status = initial_status + " - Replanning..."
            # print(self.status) # Optional print
            current_static_map = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.STATIC]
            self.map_obstacles = current_static_map

            # Store goal for planner
            self.path_planner.set_goal(goal_x, goal_y)

            new_path, nodes, parents = self.path_planner.plan_path(robot_x, robot_y, goal_x, goal_y, self.map_obstacles)

            if new_path and len(new_path) > 1:
                 self.current_planned_path = new_path
                 self.current_rrt_nodes = nodes
                 self.current_rrt_parents = parents
                 self.current_path_target_idx = 0 # Reset index for new path
                 self.last_replanning_time = current_time
                 self.status = "Replanning Successful"
            else:
                 self.status = "Replanning Failed"
                 # print(self.status) # Optional print
                 self.current_planned_path = None
                 self.current_path_target_idx = 0
                 self.current_rrt_nodes = nodes
                 self.current_rrt_parents = parents
                 # return np.array([0.0, 0.0]), self._get_controller_info() # Stop if no path - Decision moved lower

        # If still no valid path after trying to replan, prepare to stop (but check dynamic first)
        if self.current_planned_path is None or len(self.current_planned_path) <= 1:
             path_available = False
             if self.status != "Replanning Failed": # Avoid overwriting replan status
                 self.status = "No valid path - Stopping"
             # print(self.status) # Optional print
        else:
             path_available = True


        # 4. Reactive Behavior - Dynamic Obstacles (Waiting Rule)
        perceived_dynamic_obstacles = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.DYNAMIC]
        potential_velocity = self.max_velocity # Use max for conservative check

        predicted_collisions = self.waiting_rule.check_dynamic_collisions(
            robot_x, robot_y, potential_velocity, robot_orientation, perceived_dynamic_obstacles
        )

        # --- Decision Point: Wait or Continue? ---
        if self.waiting_rule.should_wait(predicted_collisions):
            # --- DATA COLLECTION TRIGGER ---
            # if self.collecting_data and path_available: # Only collect if a path exists
            #     # Calculate the action the robot WOULD take if ignoring the wait rule
            #     # Use the refactored function
            #     hypothetical_action, _ = self._calculate_path_following_action(
            #         robot_x, robot_y, robot_orientation, self.perceived_obstacles
            #     )

            #     # Store the current state (observation) and the hypothetical action
            #     # IMPORTANT: Use .copy() to avoid issues with mutable objects
            #     self.collected_data.append((observation.copy(), hypothetical_action.copy()))
            #     # print(f"Data point collected. Total: {len(self.collected_data)}") # Optional print
            # # --- END DATA COLLECTION ---

            # # Set status and return the original "wait" action
            # colliding_obs_info = [(f"ObsType({type(obs.shape).__name__})", t) for obs, t in predicted_collisions]
            # self.status = f"Waiting for dynamic obstacle(s): {colliding_obs_info}"
            # self.is_waiting = True
            # action_to_take = np.array([0.0, 0.0])

            # -- Use ML model --
            print('Using ML model to choose action')
            action_to_take = self.ml_avoidance.predict(observation)
            print(action_to_take)
            print(type(action_to_take))

        else:
            # No need to wait for dynamic obstacles
            self.is_waiting = False
            if path_available:
                # Calculate the actual action using the refactored function
                action_to_take, self.status = self._calculate_path_following_action(
                    robot_x, robot_y, robot_orientation, self.perceived_obstacles
                )
            else:
                # No path available, and not waiting for dynamic -> Stop
                action_to_take = np.array([0.0, 0.0])
                # Status should already be set (e.g., "Replanning Failed" or "No valid path")

        # Return the chosen action and controller info
        controller_info = self._get_controller_info()
        return action_to_take, controller_info

    # --- Helper functions (_get_lookahead_point, _is_path_invalidated, _normalize_angle, _slerp_angle, _get_controller_info) remain the same ---
    # ... (Paste the existing helper functions here) ...
    def _get_lookahead_point(self, robot_x, robot_y):
        # This logic remains the same as it works on the path (list of points)
        if not self.current_planned_path or len(self.current_planned_path) < 2:
            return None, self.current_path_target_idx

        robot_pos = np.array([robot_x, robot_y])
        current_target_idx = self.current_path_target_idx

        # --- Update Path Target Index ---
        search_start_idx = max(0, current_target_idx - 1)
        min_dist_to_segment_sq = float('inf')
        closest_segment_idx = current_target_idx
        projection_t = 0.0

        # Check if path is valid before accessing indices
        if not self.current_planned_path or len(self.current_planned_path) == 0:
             return None, self.current_path_target_idx

        for i in range(search_start_idx, len(self.current_planned_path) - 1):
            # Ensure indices are valid before accessing
            if i >= len(self.current_planned_path) or i + 1 >= len(self.current_planned_path):
                break # Stop if index goes out of bounds

            p1 = np.array(self.current_planned_path[i])
            p2 = np.array(self.current_planned_path[i+1])
            seg_vec = p2 - p1
            seg_len_sq = np.dot(seg_vec, seg_vec)
            if seg_len_sq < 1e-9:
                 dist_sq = np.dot(robot_pos - p1, robot_pos - p1)
                 t = 0.0
            else:
                 t = np.dot(robot_pos - p1, seg_vec) / seg_len_sq
                 t_clamped = np.clip(t, 0, 1)
                 closest_point_on_segment = p1 + t_clamped * seg_vec
                 dist_sq = np.dot(robot_pos - closest_point_on_segment, robot_pos - closest_point_on_segment)

            if dist_sq < min_dist_to_segment_sq:
                 min_dist_to_segment_sq = dist_sq
                 closest_segment_idx = i
                 projection_t = t # Unclamped t

        # Advance target index based on closest segment found
        if closest_segment_idx >= current_target_idx:
            current_target_idx = min(closest_segment_idx + 1, len(self.current_planned_path) - 1)
        elif closest_segment_idx == current_target_idx - 1 and projection_t > 1.0: # If projection past the end of segment before target
            # Check if current_target_idx is already the last index
            if current_target_idx < len(self.current_planned_path) - 1:
                current_target_idx += 1


        # --- Find Lookahead Point ---
        lookahead_point_found = None
        dist_along_path = 0.0
        start_node_idx = max(0, current_target_idx - 1)

        # Again, check path validity
        if not self.current_planned_path or len(self.current_planned_path) == 0:
            return None, self.current_path_target_idx
        # Ensure start_node_idx is valid
        start_node_idx = min(start_node_idx, len(self.current_planned_path) - 1)


        # Calculate distance from robot to the effective start point on the path for lookahead search
        start_search_point_on_path = np.array(self.current_planned_path[start_node_idx])
        if start_node_idx < len(self.current_planned_path) -1:
            p1 = np.array(self.current_planned_path[start_node_idx])
            p2 = np.array(self.current_planned_path[start_node_idx+1])
            seg_vec = p2 - p1
            seg_len_sq = np.dot(seg_vec, seg_vec)
            if seg_len_sq > 1e-9:
                t = np.dot(robot_pos - p1, seg_vec) / seg_len_sq
                t_clamped = np.clip(t, 0, 1)
                start_search_point_on_path = p1 + t_clamped * seg_vec # Closest point on path segment

        # Calculate remaining distance from the 'start_search_point_on_path' along the path
        current_dist_on_path = 0.0
        for i in range(start_node_idx, len(self.current_planned_path) - 1):
            # Ensure indices are valid
            if i >= len(self.current_planned_path) or i + 1 >= len(self.current_planned_path):
                break

            p1 = np.array(self.current_planned_path[i])
            p2 = np.array(self.current_planned_path[i+1])
            segment_vec = p2 - p1
            segment_len = np.linalg.norm(segment_vec)

            if segment_len < 1e-6: continue

            # How much of this segment is relevant for lookahead search?
            dist_to_p1 = np.linalg.norm(start_search_point_on_path - p1)
            # dist_to_p2 = np.linalg.norm(start_search_point_on_path - p2) # Not strictly needed

            if i == start_node_idx: # Robot projection is on or before this segment
                dist_remaining_on_segment = segment_len - np.clip(dist_to_p1, 0, segment_len)
            else: # Robot is on a previous segment
                dist_remaining_on_segment = segment_len

            if current_dist_on_path + dist_remaining_on_segment >= self.lookahead_distance:
                # Lookahead point is on this segment
                needed_dist_on_this_segment = self.lookahead_distance - current_dist_on_path
                # Calculate the fraction needed along the *full* segment length
                ratio = needed_dist_on_this_segment / segment_len if segment_len > 1e-6 else 0
                ratio = np.clip(ratio, 0, 1) # Clip ratio to handle edge cases/numerical issues
                lookahead_point_found = tuple(p1 + ratio * segment_vec)
                break
            else:
                current_dist_on_path += dist_remaining_on_segment
                start_search_point_on_path = p2 # Advance the reference point

        if lookahead_point_found is None and self.current_planned_path: # Check path not empty
             lookahead_point_found = self.current_planned_path[-1] # Use last point if path too short or lookahead extends beyond path

        # Ensure current_target_idx is valid before returning
        if not self.current_planned_path or len(self.current_planned_path) == 0:
             current_target_idx = 0
        else:
             current_target_idx = min(current_target_idx, len(self.current_planned_path) - 1)

        return lookahead_point_found, current_target_idx


    def _is_path_invalidated(self, robot_x, robot_y):
        # Uses the planner's segment check which now handles Obstacle objects
        if not self.current_planned_path or len(self.current_planned_path) < 2 or self.current_path_target_idx >= len(self.current_planned_path) - 1:
             return False

        # Find closest segment index to robot (reuse logic from lookahead)
        robot_pos = np.array([robot_x, robot_y])
        min_dist_sq = float('inf')
        closest_segment_idx = self.current_path_target_idx # Initialize
        search_start_idx = max(0, self.current_path_target_idx - 1)
        # Ensure search_start_idx is valid
        search_start_idx = min(search_start_idx, len(self.current_planned_path) - 2) if len(self.current_planned_path) > 1 else 0


        for i in range(search_start_idx, len(self.current_planned_path) - 1):
             p1 = np.array(self.current_planned_path[i])
             p2 = np.array(self.current_planned_path[i+1])
             seg_vec = p2 - p1
             seg_len_sq = np.dot(seg_vec, seg_vec)
             if seg_len_sq < 1e-9: dist_sq = np.dot(robot_pos - p1, robot_pos - p1)
             else:
                  t = np.clip(np.dot(robot_pos - p1, seg_vec) / seg_len_sq, 0, 1)
                  closest_point = p1 + t * seg_vec
                  dist_sq = np.dot(robot_pos - closest_point, robot_pos - closest_point)
             if dist_sq < min_dist_sq:
                  min_dist_sq = dist_sq
                  closest_segment_idx = i

        # Check segments ahead
        start_check_idx = closest_segment_idx
        end_check_idx = min(len(self.current_planned_path) - 2, start_check_idx + self.path_invalidation_check_horizon)

        for i in range(start_check_idx, end_check_idx + 1):
             # Ensure indices are valid
             if i >= len(self.current_planned_path) -1: break

             p1 = self.current_planned_path[i]
             p2 = self.current_planned_path[i+1]
             # Check this path segment against *currently perceived obstacles*
             for obs in self.perceived_obstacles:
                  if obs.intersects_segment(p1, p2, self.robot_radius):
                    #   print(f"Path invalidated: Segment {i} blocked by perceived {type(obs.shape).__name__} near ({obs.x:.1f},{obs.y:.1f})")
                      return True # Path is blocked ahead

        return False

    def _normalize_angle(self, angle):
        while angle > np.pi: angle -= 2 * np.pi
        while angle < -np.pi: angle += 2 * np.pi
        return angle

    def _slerp_angle(self, a1, a2, t):
        a1_norm = self._normalize_angle(a1)
        a2_norm = self._normalize_angle(a2)
        diff = self._normalize_angle(a2_norm - a1_norm)
        # Handle potential large angle differences correctly with SLERP concept
        # For angles, linear interpolation is usually sufficient if normalized correctly
        # If using true SLERP: dot = np.cos(a1_norm) * np.cos(a2_norm) + np.sin(a1_norm) * np.sin(a2_norm)
        # theta = np.arccos(np.clip(dot, -1, 1)) etc. - linear is simpler here:
        return self._normalize_angle(a1_norm + diff * t)


    def _get_controller_info(self):
        return {
            "status": self.status,
            "planned_path": self.current_planned_path,
            "rrt_nodes": self.current_rrt_nodes,
            "rrt_parents": self.current_rrt_parents,
            "target_idx": self.current_path_target_idx,
            "is_waiting": self.is_waiting # Add waiting status for debugging/info
            # "rays": self.last_ray_viz_points (if rays implemented fully)
            }