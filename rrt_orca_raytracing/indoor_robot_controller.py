# --- START OF FILE indoor_robot_controller.py ---

# indoor_robot_controller.py

import time
import math
import numpy as np
from indoor_robot_env import IndoorRobotEnv # Assuming this exists
from components.obstacle import Obstacle, ObstacleType, StaticObstacle, DynamicObstacle # Import specific types if needed
from components.shape import Circle, Rectangle, Shape
from utils.ray_tracing_algorithm import RayTracingAlgorithm
from utils.rrt_planner import RRTPathPlanner
# Remove or comment out WaitingRule if ORCA completely replaces its action logic
# from utils.waiting_rule import WaitingRule
from utils.obstacle_identifier import ObstacleIdentifier
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
        self.ray_tracer = RayTracingAlgorithm(self.width, self.height, self.robot_radius)
        # self.waiting_rule = WaitingRule(self.robot_radius) # Keep if needed for detection, disable if ORCA handles all
        self.path_planner = RRTPathPlanner(self.width, self.height, self.robot_radius)

        # Controller parameters (mostly same)
        self.goal_threshold = self.robot_radius + 5
        self.max_velocity = self.action_space.high[0]
        self.min_velocity = 0.2 # Keep minimum linear velocity? ORCA might output zero.
        self.obstacle_slow_down_distance = self.robot_radius * 5
        self.obstacle_avoid_distance = self.robot_radius * 4 # Used for static avoidance blending
        self.lookahead_distance = self.robot_radius * 2
        self.path_invalidation_check_horizon = 5

        # --- ORCA Parameters ---
        self.orca_tau = 2.0 # Prediction horizon for ORCA (tune this)
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
         self.perceived_obstacles = []
         # --- Reset Obstacle Memory ---
         self.discovered_static_obstacles = []
         self.map_obstacles = []
         self.known_obstacle_descriptions = set()
         # --- End Reset Obstacle Memory ---
        #  self.is_waiting = False
         self.last_replanning_time = -np.inf
         self.status = "Reset"
        #  self.pre_action = np.array([0.0, 0.0])
         # Do NOT reset collected_data here

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
        # ORCA works best with a preferred velocity vector. Pointing towards the lookahead point is better.
        lookahead_point, _ = self._get_lookahead_point(robot_x, robot_y) # Use last known pos

        if lookahead_point is None or not self.current_planned_path:
             # Fallback: Use current orientation or stop if no path
             pref_vx = linear_velocity * np.cos(robot_orientation)
             pref_vy = linear_velocity * np.sin(robot_orientation)
        else:
             target_vec = np.array(lookahead_point) - np.array([robot_x, robot_y])
             dist = np.linalg.norm(target_vec)
             if dist < 1e-6:
                  # Very close to lookahead, use current orientation or aim for next point if possible
                  pref_vx = linear_velocity * np.cos(robot_orientation)
                  pref_vy = linear_velocity * np.sin(robot_orientation)
             else:
                  # Point the preferred velocity towards lookahead, scaled by intended linear speed
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


    # --- ( _calculate_path_following_action remains largely the same, handles STATIC obstacles ) ---
    def _calculate_path_following_action(self, robot_x, robot_y, robot_orientation, perceived_static_obstacles): # Now only receives static
        """ Calculates the action based on path following and perceived STATIC obstacles """
        # Returns: action = np.array([velocity, steering_angle]), calculation_status

        # --- Start logic (mostly copied, ensure it uses perceived_static_obstacles) ---
        calculation_status = "Following Path (Calculating)"
        if not self.current_planned_path: # Check if path exists before getting lookahead
            calculation_status = "Path Error (No Path) - Stopping"
            return np.array([0.0, 0.0]), calculation_status

        lookahead_point, self.current_path_target_idx = self._get_lookahead_point(robot_x, robot_y)

        if lookahead_point is None:
            calculation_status = "Path Error (Lookahead) - Stopping"
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

        # --- Reactive Static Obstacle Avoidance (using perceived_static_obstacles) ---
        final_target_orientation = target_orientation
        min_dist_to_static_boundary = float('inf')
        avoiding_obstacle = None
        current_min_avoid_eff_dist = float('inf')

        for obstacle in perceived_static_obstacles: # Iterate only over static ones passed
            obs_pos = np.array(obstacle.get_position())
            robot_pos = np.array([robot_x, robot_y])

            # Use efficient distance calculation (center-to-center minus radii/half-widths)
            # eff_dist = obstacle.shape.get_efficient_distance(robot_x, robot_y, obs_pos[0], obs_pos[1]) - self.robot_radius
            eff_vec = obstacle.shape.get_effective_vector(robot_x, robot_y, obs_pos[0], obs_pos[1])
            eff_dist = np.linalg.norm(eff_vec) # Effective distance
            # Simplified distance for avoidance trigger: center-to-center distance
            # center_dist = np.linalg.norm(robot_pos - obs_pos)
            # Approximate effective distance considering robot radius and obstacle size for trigger
            # eff_dist = center_dist - self.robot_radius - obstacle.shape.get_effective_radius()
            min_dist_to_static_boundary = min(min_dist_to_static_boundary, eff_dist)


            if eff_dist < self.obstacle_avoid_distance:
                vec_to_obs = eff_vec
                # Check if obstacle is roughly in front based on target direction
                angle_to_obs = np.arctan2(vec_to_obs[1], vec_to_obs[0])
                angle_diff_to_target = abs(self._normalize_angle(angle_to_obs - target_orientation))

                # # Avoidance cone
                if angle_diff_to_target < np.pi * 2: # 90 degree cone relative to target dir
                    # Choose the closest one within the cone based on eff_dist
                    if avoiding_obstacle is None or eff_dist < current_min_avoid_eff_dist:
                            avoiding_obstacle = obstacle
                            current_min_avoid_eff_dist = eff_dist # Update min distance for chosen one

        if avoiding_obstacle is not None:
            final_avoid_eff_dist = current_min_avoid_eff_dist
            calculation_status = f"Avoiding static {type(avoiding_obstacle.shape).__name__} (Calculating)"
            print(calculation_status) # Debug print
            # Use ray tracing for avoidance direction
            avoidance_orientation = self.ray_tracer.avoid_static_obstacle(
                robot_x, robot_y, self.robot_radius, robot_orientation, avoiding_obstacle, target_x, target_y
            )
            # Blend factor based on proximity
            blend_factor = 1.0 - np.clip(final_avoid_eff_dist / self.obstacle_avoid_distance, 0.0, 1.0) ** 4

            print (f"Blend factor: {blend_factor:.2f}") # Debug print
            # blend_factor = blend_factor**2 # Make blending stronger when close
            final_target_orientation = self._slerp_angle(target_orientation, avoidance_orientation, blend_factor)
            print(f'target_orientation: {target_orientation:.2f}, avoidance_orientation: {avoidance_orientation:.2f}, final_target_orientation: {final_target_orientation:.2f}') # Debug print

        # --- Calculate Control Action ---
        steering_angle = self._normalize_angle(final_target_orientation - robot_orientation)
        steering_angle = np.clip(steering_angle, self.action_space.low[1], self.action_space.high[1])

        velocity = self.max_velocity
        # --- Velocity Scaling ---
        # Factor based on steering magnitude
        max_steer = abs(self.action_space.high[1]) if self.action_space.high[1] > 1e-6 else np.pi # Avoid division by zero
        norm_steer_mag = abs(steering_angle) / max_steer
        steering_vel_factor = max(0.2, np.cos(norm_steer_mag * np.pi / 2)**0.5)

        # Factor based on proximity to closest PERCEIVED static obstacle boundary
        proximity_vel_factor = 1.0
        if min_dist_to_static_boundary < self.obstacle_avoid_distance:
             proximity_vel_factor = np.clip(min_dist_to_static_boundary / self.obstacle_avoid_distance, 0.1, 1.0)**0.5
        elif min_dist_to_static_boundary < self.obstacle_slow_down_distance:
            slow_down_ratio = (min_dist_to_static_boundary - self.obstacle_avoid_distance) / (self.obstacle_slow_down_distance - self.obstacle_avoid_distance)
            proximity_vel_factor = 0.6 + 0.4 * slow_down_ratio
            proximity_vel_factor = np.clip(proximity_vel_factor, 0.6, 1.0)

        velocity = self.max_velocity * steering_vel_factor * proximity_vel_factor
        velocity = np.clip(velocity, self.min_velocity, self.max_velocity) # Ensure min/max velocity

        action = np.array([velocity, steering_angle], dtype=np.float32)

        # Update status if not already set to avoiding/error
        if "Calculating" in calculation_status:
             if np.linalg.norm(action) > 1e-6 or self.current_planned_path:
                 calculation_status = "Following Path (Static Avoidance)"
             else:
                 calculation_status = "Path Error (No Action)"

        return action, calculation_status
        # --- End logic ---


    def get_action(self, observation: dict):
        # --- 1. Perceive Environment & Update Obstacle Memory ---
        robot_x, robot_y, robot_velocity, robot_orientation, goal_x, goal_y = observation['base_observation']
        # Identify currently visible obstacles
        self.perceived_obstacles = self.obstacle_identifier.identify(observation)
        # Update the memory of static obstacles
        memory_updated = self._update_obstacle_memory(self.perceived_obstacles)

        # Separate perceived obstacles for different uses
        perceived_static_obstacles = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.STATIC]
        perceived_dynamic_obstacles = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.DYNAMIC]

        # Estimate robot's current world velocity [vx, vy]
        current_robot_velocity_vector = robot_velocity * np.array([np.cos(robot_orientation), np.sin(robot_orientation)])

        # --- 2. Check Goal Reached ---
        distance_to_goal = np.linalg.norm(np.array([robot_x, robot_y]) - np.array([goal_x, goal_y]))
        if distance_to_goal < self.goal_threshold:
            self.status = "Goal Reached - Stopping"
            self.current_planned_path = None
            self.current_path_target_idx = 0
            info = self._get_controller_info()
            info['goal_reached'] = True
            final_action = np.array([0.0, 0.0]) # Ensure stop action
            # --- Data Collection (for final step) ---
            if self.collecting_data:
                 # Ensure data format is consistent (e.g., velocity vector if used for training)
                 # Here we store observation and the [lin_vel, steer_angle] action
                 self.collected_data.append((observation.copy(), final_action.copy()))
            return final_action, info

        # --- 3. Replanning Logic ---
        needs_replan = False
        initial_status = "Checking Path..."
        if self.current_planned_path is None or len(self.current_planned_path) <= 1:
             initial_status = "No valid path"
             needs_replan = True
        elif self.current_path_target_idx >= len(self.current_planned_path):
             initial_status = "Reached end of path"
             needs_replan = True # Replan to confirm goal or if stuck
             if distance_to_goal > self.goal_threshold * 1.5:
                 needs_replan = True
             else:
                 needs_replan = False
                 self.status = "Near Goal (End of Path)"
        # --- Check path invalidation using DISCOVERED static obstacles ---
        # elif self._is_path_invalidated(robot_x, robot_y, self.discovered_static_obstacles): # Check against memory
        #      initial_status = "Path invalidated by obstacle memory"
        #      needs_replan = True
        # --- Check path invalidation using PERCEIVED obstacles (static and dynamic) ---
        elif self._is_path_invalidated(robot_x, robot_y, self.perceived_obstacles): # Check against current perception
              initial_status = "Path invalidated by perceived obstacle"
              needs_replan = True
        # --- Force replan if static obstacle memory was updated ---
        # elif memory_updated:
        #      initial_status = "Static obstacle memory updated"
        #      needs_replan = True


        if needs_replan:
            self.status = initial_status + " - Replanning..."
            # print(self.status) # Optional print
            # --- Use the DISCOVERED static obstacles for planning ---
            self.map_obstacles = list(self.discovered_static_obstacles) # Update map obstacles used for planning

            self.path_planner.set_goal(goal_x, goal_y) # Ensure goal is set

            # --- Pass the DISCOVERED obstacles to the planner ---
            new_path, nodes, parents = self.path_planner.plan_path(
                robot_x, robot_y, goal_x, goal_y, self.map_obstacles, smooth_path=True
            )

            if new_path and len(new_path) > 1:
                 self.current_planned_path = new_path
                 self.current_rrt_nodes = nodes
                 self.current_rrt_parents = parents
                 self.current_path_target_idx = 0 # Reset index for new path
                 self.status = "Replanning Successful"
            else:
                 self.status = "Replanning Failed"
                 # print(self.status) # Optional print
                 self.current_planned_path = None # Clear failed path
                 self.current_path_target_idx = 0
                 self.current_rrt_nodes = nodes # Keep tree for viz
                 self.current_rrt_parents = parents

        # Check path availability *after* potential replanning
        path_available = self.current_planned_path is not None and len(self.current_planned_path) > 1

        # --- 4. Calculate Preferred Velocity (based on path following & PERCEIVED static avoidance) ---
        if not path_available:
            preferred_action = np.array([0.0, 0.0])
            if "Replanning Failed" not in self.status: self.status = "No Path - Stopping"
        else:
            # Calculate the action considering path following and only PERCEIVED static obstacles for immediate reaction
            preferred_action, self.status = self._calculate_path_following_action(
                robot_x, robot_y, robot_orientation, perceived_static_obstacles # Pass only currently seen static obs
            )

        # Convert preferred action (linear_vel, steering) into a preferred world velocity vector [vx, vy]
        preferred_velocity_vector = self._action_to_velocity_vector(
            preferred_action[0], preferred_action[1], robot_orientation, robot_x, robot_y
        )


        # --- 5. Prepare ORCA Inputs ---
        # 5.1 Robot Agent
        robot_agent = OrcaAgent(
            position=np.array([robot_x, robot_y]),
            velocity=current_robot_velocity_vector, # Use the estimated velocity
            radius=self.robot_radius,
            max_speed=self.max_velocity,
            pref_velocity=preferred_velocity_vector
        )

        # 5.2 Dynamic Obstacle Agents (use PERCEIVED dynamic obstacles)
        dynamic_orca_agents = []
        for obs in perceived_dynamic_obstacles:
            obs_pos = np.array(obs.get_position())
            direction_norm = np.linalg.norm(obs.direction)
            if direction_norm > 1e-6:
                 norm_direction = obs.direction / direction_norm
            else:
                 norm_direction = np.array([0.0, 0.0])

            obs_velocity_vector = obs.velocity * norm_direction
            effective_radius = self._get_orca_effective_radius(obs.shape)

            collider_agent = OrcaAgent(
                position=obs_pos,
                velocity=obs_velocity_vector,
                radius=effective_radius,
                max_speed=obs.velocity * 1.5, # Allow headroom
                pref_velocity=obs_velocity_vector
            )
            dynamic_orca_agents.append(collider_agent)

        # 5.3 Time parameters
        t_horizon = self.orca_tau
        dt_step = self.env_dt

        # --- 6. Call ORCA Algorithm ---
        optimal_velocity_vector = preferred_velocity_vector # Default to preferred velocity
        orca_status_suffix = "" # To append to status if ORCA runs

        if dynamic_orca_agents: # Only call ORCA if dynamic obstacles are present
            try:
                orca_optimal_velocity, _ = calculate_orca_velocity(
                    agent=robot_agent,
                    colliding_agents=dynamic_orca_agents,
                    t=t_horizon,
                    dt=dt_step
                )

                # Limit the magnitude of the ORCA velocity to max speed
                orca_speed = np.linalg.norm(orca_optimal_velocity)
                if orca_speed > self.max_velocity:
                     orca_optimal_velocity = (orca_optimal_velocity / orca_speed) * self.max_velocity

                if np.isnan(orca_optimal_velocity).any():
                     print("Warning: ORCA returned NaN! Using preferred velocity instead.")
                     # Fallback to preferred velocity might be safer than stopping if path exists
                     optimal_velocity_vector = preferred_velocity_vector
                     orca_status_suffix = " (ORCA NaN Fallback)"
                else:
                     optimal_velocity_vector = orca_optimal_velocity # Use ORCA result
                     orca_status_suffix = " (ORCA Active)"

            except InfeasibleError:
                print("Warning: ORCA found no feasible solution! Stopping.")
                optimal_velocity_vector = np.array([0.0, 0.0]) # Fallback: Stop
                orca_status_suffix = " (ORCA Infeasible - Stopping)"
            except Exception as e:
                print(f"Error during ORCA calculation: {e}. Stopping.")
                optimal_velocity_vector = np.array([0.0, 0.0]) # Fallback: Stop
                orca_status_suffix = f" (ORCA Error - Stopping)"

            # --- Data Collection (for dynamic avoidance steps) ---
            # Decide what to store: observation + final action ([lin, steer]) or obs + optimal_vel_vec?
            # Let's store obs + optimal_velocity_vector as it represents the core ORCA output
            if self.collecting_data:
                 self.collected_data.append((observation.copy(), optimal_velocity_vector.copy()))
                 # If you prefer action:
                 # final_action_for_data = self._velocity_vector_to_action(optimal_velocity_vector, robot_orientation)
                 # self.collected_data.append((observation.copy(), final_action_for_data.copy()))

        # Update status based on whether ORCA ran and the outcome
        if orca_status_suffix:
             # If ORCA failed and caused stop, overwrite status
             if "Stopping" in orca_status_suffix:
                  self.status = orca_status_suffix.strip() # Remove leading space if any
             else:
                  # Append ORCA info to the existing status (e.g., "Following Path (Static Avoidance) (ORCA Active)")
                  # Avoid appending if status already indicates an error/stop condition
                  if "Error" not in self.status and "Stopping" not in self.status and "Failed" not in self.status:
                     self.status += orca_status_suffix
        else:
             # ORCA didn't run (no dynamic obstacles)
             if path_available and np.linalg.norm(preferred_action)>1e-6:
                 if "Avoiding static" not in self.status: # Don't overwrite avoidance status
                     self.status = "Following Path (No dynamic obstacles)"
             elif not path_available:
                 if "Replanning Failed" not in self.status: # Keep failure status
                     self.status = "No Path / Stopped"


        # --- 7. Convert Final Velocity Output to Robot Action ---
        final_action = self._velocity_vector_to_action(optimal_velocity_vector, robot_orientation)

        # --- 8. Return Action and Info ---
        controller_info = self._get_controller_info()
        controller_info["status"] = self.status # Ensure latest status is in info
        # Add discovered obstacles to info for visualization/debugging
        controller_info["discovered_static_obstacles"] = self.discovered_static_obstacles
        controller_info["map_obstacles_used_for_plan"] = self.map_obstacles # Show what was actually used

        # self.pre_action = final_action.copy()

        return final_action, controller_info


    # --- Helper functions (_get_lookahead_point, _is_path_invalidated, _normalize_angle, _slerp_angle, _get_controller_info) ---
    # Paste the existing helper functions here (make sure they are indented correctly within the class)
    # ... (Keep _get_lookahead_point as is) ...
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
        search_start_idx = max(0, current_target_idx - 2) # Look back a bit more
        search_end_idx = min(len(self.current_planned_path) - 2, current_target_idx + 5) # Look ahead a bit
        min_dist_to_segment_sq = float('inf')
        closest_segment_idx = current_target_idx # Default
        projection_t = 0.0 # Projection factor onto the closest segment

        if len(self.current_planned_path) < 2: # Path is just one point
             dist_to_target_sq = np.sum((robot_pos - np.array(self.current_planned_path[0]))**2)
             if dist_to_target_sq < (self.lookahead_distance * 0.5)**2:
                 current_target_idx = 0
             return self.current_planned_path[0], 0


        for i in range(search_start_idx, search_end_idx + 1):
            if i >= len(self.current_planned_path) - 1: break

            p1 = np.array(self.current_planned_path[i])
            p2 = np.array(self.current_planned_path[i+1])
            seg_vec = p2 - p1
            seg_len_sq = np.dot(seg_vec, seg_vec)

            if seg_len_sq < 1e-12: # Segment is effectively a point
                 dist_sq = np.sum((robot_pos - p1)**2)
                 t = 0.0
            else:
                 t = np.dot(robot_pos - p1, seg_vec) / seg_len_sq
                 t_clamped = np.clip(t, 0, 1)
                 closest_point_on_segment = p1 + t_clamped * seg_vec
                 dist_sq = np.sum((robot_pos - closest_point_on_segment)**2)

            if dist_sq < min_dist_to_segment_sq:
                 min_dist_to_segment_sq = dist_sq
                 closest_segment_idx = i
                 projection_t = t # Store the unclamped projection factor

        # --- Advance Target Index ---
        if closest_segment_idx == current_target_idx and projection_t > 0.1:
             if current_target_idx < len(self.current_planned_path) - 1:
                  current_target_idx += 1
        elif closest_segment_idx > current_target_idx:
              current_target_idx = min(closest_segment_idx + 1, len(self.current_planned_path) - 1)
        elif closest_segment_idx == current_target_idx - 1 and projection_t > 0.5 :
             if current_target_idx < len(self.current_planned_path) -1:
                 current_target_idx += 1


        current_target_idx = min(max(0, current_target_idx), len(self.current_planned_path) - 1)
        self.current_path_target_idx = current_target_idx # Update controller state


        # --- Find Lookahead Point ---
        search_start_node_idx = max(0, current_target_idx -1) # Start from segment beginning at this node index
        start_measure_point = robot_pos # Default
        if len(self.current_planned_path) >= 2:
             p1_search = np.array(self.current_planned_path[search_start_node_idx])
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

        cumulative_dist = 0.0
        lookahead_point_found = None
        first_seg_end_idx = min(search_start_node_idx + 1, len(self.current_planned_path)-1)
        dist_on_first_segment = np.linalg.norm(np.array(self.current_planned_path[first_seg_end_idx]) - start_measure_point)

        if dist_on_first_segment >= self.lookahead_distance:
            p1 = start_measure_point
            p2 = np.array(self.current_planned_path[first_seg_end_idx])
            vec = p2 - p1
            norm_vec = vec / dist_on_first_segment if dist_on_first_segment > 1e-6 else vec
            lookahead_point_found = tuple(p1 + norm_vec * self.lookahead_distance)
        else:
            cumulative_dist += dist_on_first_segment
            for i in range(search_start_node_idx + 1, len(self.current_planned_path) - 1):
                p1 = np.array(self.current_planned_path[i])
                p2 = np.array(self.current_planned_path[i+1])
                segment_vec = p2 - p1
                segment_len = np.linalg.norm(segment_vec)

                if cumulative_dist + segment_len >= self.lookahead_distance:
                    remaining_dist = self.lookahead_distance - cumulative_dist
                    ratio = remaining_dist / segment_len if segment_len > 1e-6 else 0
                    lookahead_point_found = tuple(p1 + ratio * segment_vec)
                    break
                else:
                    cumulative_dist += segment_len

        if lookahead_point_found is None:
            lookahead_point_found = tuple(self.current_planned_path[-1]) # Target the end

        return lookahead_point_found, self.current_path_target_idx

    # Modify _is_path_invalidated to accept the list of obstacles to check against
    def _is_path_invalidated(self, robot_x, robot_y, obstacles_to_check: list):
        if not self.current_planned_path or len(self.current_planned_path) < 2:
             return False

        robot_pos = np.array([robot_x, robot_y])

        # --- Find segment closest to robot ---
        min_dist_sq = float('inf')
        closest_segment_idx = 0
        search_center = self.current_path_target_idx
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
                    #    print(f"Path invalidated: Segment {i} blocked by {obs.type.name} obstacle") # Debug
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
            "target_idx": self.current_path_target_idx,
            # Add obstacle memory info for debugging/visualization
            "discovered_static_obstacles": [obs for obs in self.discovered_static_obstacles], # Return a copy
            "map_obstacles_used_for_plan": [obs for obs in self.map_obstacles], # Return a copy
            }

# --- END OF FILE indoor_robot_controller.py ---