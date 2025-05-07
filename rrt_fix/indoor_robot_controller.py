import time
import math
import numpy as np
from indoor_robot_env import IndoorRobotEnv
from components.obstacle import Obstacle, ObstacleType
from components.shape import Circle, Rectangle
from utils.ray_tracing_algorithm import RayTracingAlgorithm # Still used for reactive avoidance
from utils.rrt_planner import RRTPathPlanner
from utils.waiting_rule import WaitingRule
from utils.obstacle_identifier import ObstacleIdentifier

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
        self.ray_tracer = RayTracingAlgorithm(self.width, self.height, self.robot_radius)
        self.waiting_rule = WaitingRule(self.robot_radius)
        self.path_planner = RRTPathPlanner(self.width, self.height, self.robot_radius)

        # Controller parameters
        self.goal_threshold = self.robot_radius + 5
        self.max_velocity = self.action_space.high[0]
        self.min_velocity = 0.3
        self.obstacle_slow_down_distance = self.robot_radius * 5
        self.obstacle_avoid_distance = self.robot_radius * 3
        # self.lookahead_distance = self.robot_radius * 4 # REMOVED - Pure pursuit parameter
        self.path_invalidation_check_horizon = 5
        self.waypoint_reached_threshold = self.robot_radius * 1.5 # ADDED - For waypoint following

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

    # New method for simple waypoint following
    def _update_path_target_and_get_waypoint(self, robot_x, robot_y):
        if not self.current_planned_path or len(self.current_planned_path) == 0:
            return None, self.current_path_target_idx

        # Ensure current_path_target_idx is within valid bounds.
        # This can be important if the path changes (e.g., becomes shorter after replanning).
        if self.current_path_target_idx >= len(self.current_planned_path):
            self.current_path_target_idx = len(self.current_planned_path) - 1
        
        # If path has only one point (e.g., start is very near goal, or path is just the goal itself),
        # target that single point. The main goal check in get_action handles actual goal reaching.
        if len(self.current_planned_path) == 1:
            self.current_path_target_idx = 0 # Ensure it's 0 for a single-point path
            return self.current_planned_path[0], self.current_path_target_idx

        current_target_waypoint_coords = self.current_planned_path[self.current_path_target_idx]
        robot_pos = np.array([robot_x, robot_y])
        dist_to_current_wp = np.linalg.norm(robot_pos - np.array(current_target_waypoint_coords))

        # If the robot is close enough to the current target waypoint,
        # and it's not the final waypoint in the path, advance the target index.
        if dist_to_current_wp < self.waypoint_reached_threshold and \
           self.current_path_target_idx < len(self.current_planned_path) - 1:
            self.current_path_target_idx += 1
        
        # The target for this control step is the (potentially updated) waypoint.
        final_target_waypoint_coords = self.current_planned_path[self.current_path_target_idx]
        
        return final_target_waypoint_coords, self.current_path_target_idx


    def get_action(self, observation: np.ndarray):
        current_time = time.time()
        self.status = "Processing"

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
        if self.current_planned_path is None or len(self.current_planned_path) == 0:
             self.status = "No valid path"
             needs_replan = True
        elif self.current_path_target_idx >= len(self.current_planned_path): 
             self.status = "Target index out of bounds for current path"
             needs_replan = True 
             self.current_planned_path = None 
        elif self._is_path_invalidated(robot_x, robot_y):
             self.status = "Path invalidated by new obstacle"
             needs_replan = True

        if needs_replan:
            self.status += " - Replanning..."
            print(self.status)
            current_static_map = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.STATIC]
            self.map_obstacles = current_static_map

            new_path, nodes, parents = self.path_planner.plan_path(robot_x, robot_y, goal_x, goal_y, self.map_obstacles)

            if new_path and len(new_path) > 0: 
                 self.current_planned_path = new_path
                 self.current_rrt_nodes = nodes
                 self.current_rrt_parents = parents
                 self.current_path_target_idx = 0 
                 self.last_replanning_time = current_time
                 self.status = "Replanning Successful"
            else:
                 self.status = "Replanning Failed"
                 print(self.status)
                 self.current_planned_path = None
                 self.current_path_target_idx = 0
                 self.current_rrt_nodes = nodes 
                 self.current_rrt_parents = parents
                 return np.array([0.0, 0.0]), self._get_controller_info() 

        if self.current_planned_path is None or len(self.current_planned_path) == 0:
             self.status = "No valid path - Stopping"
             return np.array([0.0, 0.0]), self._get_controller_info()


        # 4. Reactive Behavior - Dynamic Obstacles (Waiting Rule)
        perceived_dynamic_obstacles = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.DYNAMIC]
        potential_velocity = self.max_velocity 

        predicted_collisions = self.waiting_rule.check_dynamic_collisions(
            robot_x, robot_y, potential_velocity, robot_orientation, perceived_dynamic_obstacles
        )

        if self.waiting_rule.should_wait(predicted_collisions):
             colliding_obs_info = [(f"ObsType({type(obs.shape).__name__})", t) for obs, t in predicted_collisions]
             self.status = f"Waiting for dynamic obstacle(s): {colliding_obs_info}"
             self.is_waiting = True
             return np.array([0.0, 0.0]), self._get_controller_info() 
        else:
             self.is_waiting = False


        # 5. Path Following (Waypoint-based) & Static Obstacle Avoidance
        self.status = "Following Path"
        # MODIFIED: Call new method for waypoint following
        target_waypoint_coords, self.current_path_target_idx = \
            self._update_path_target_and_get_waypoint(robot_x, robot_y)

        if target_waypoint_coords is None:
             self.status = "Path Error (No Target Waypoint) - Stopping"
             print(self.status)
             self.current_planned_path = None 
             return np.array([0.0, 0.0]), self._get_controller_info()

        target_x, target_y = target_waypoint_coords 
        target_vector = np.array([target_x - robot_x, target_y - robot_y])
        target_distance = np.linalg.norm(target_vector)

        if target_distance < 1e-6: # Robot is (almost) exactly at the current target waypoint
             if self.current_path_target_idx < len(self.current_planned_path) - 1:
                  # Aim for the *next* waypoint in the path to ensure progress if not already advanced.
                  next_actual_target_coords = self.current_planned_path[self.current_path_target_idx + 1]
                  target_vector = np.array([next_actual_target_coords[0] - robot_x, next_actual_target_coords[1] - robot_y])
                  if np.linalg.norm(target_vector) > 1e-6:
                       target_orientation = np.arctan2(target_vector[1], target_vector[0])
                  else: 
                       target_orientation = robot_orientation 
             else:
                  # Robot is at the final waypoint of the path. Maintain current heading.
                  target_orientation = robot_orientation
        else:
             target_orientation = np.arctan2(target_vector[1], target_vector[0])

        # --- Reactive Static Obstacle Avoidance (logic remains the same) ---
        final_target_orientation = target_orientation
        min_dist_to_static_boundary = float('inf')
        avoiding_obstacle = None
        current_min_avoid_eff_dist_for_candidate = float('inf')

        perceived_static_obstacles = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.STATIC]
        for obstacle in perceived_static_obstacles:
             eff_dist = obstacle.shape.get_efficient_distance(robot_x, robot_y, *obstacle.get_position()) - self.robot_radius
             min_dist_to_static_boundary = min(min_dist_to_static_boundary, eff_dist)

             if eff_dist < self.obstacle_avoid_distance:
                  vec_to_obs = obstacle.get_position() - np.array([robot_x, robot_y])
                  angle_to_obs = np.arctan2(vec_to_obs[1], vec_to_obs[0])
                  angle_diff_to_target = abs(self._normalize_angle(angle_to_obs - target_orientation))

                  if angle_diff_to_target < np.pi / 1.9: 
                      if avoiding_obstacle is None or eff_dist < current_min_avoid_eff_dist_for_candidate:
                          avoiding_obstacle = obstacle
                          current_min_avoid_eff_dist_for_candidate = eff_dist


        if avoiding_obstacle is not None:
            final_avoid_eff_dist = avoiding_obstacle.shape.get_efficient_distance(robot_x, robot_y, *avoiding_obstacle.get_position()) - self.robot_radius
            self.status = f"Avoiding static {type(avoiding_obstacle.shape).__name__}"
            avoidance_orientation = self.ray_tracer.avoid_static_obstacle(
                robot_x, robot_y, self.robot_radius, robot_orientation, avoiding_obstacle, target_x, target_y
            )
            blend_factor = 1.0 - np.clip(final_avoid_eff_dist / self.obstacle_avoid_distance, 0.0, 1.0)
            blend_factor = blend_factor**2
            final_target_orientation = self._slerp_angle(target_orientation, avoidance_orientation, blend_factor)


        # 6. Calculate Control Action (logic remains largely the same)
        steering_angle = self._normalize_angle(final_target_orientation - robot_orientation)
        steering_angle = np.clip(steering_angle, self.action_space.low[1], self.action_space.high[1])

        velocity = self.max_velocity
        max_steer = abs(self.action_space.high[1])
        norm_steer_mag = abs(steering_angle) / max_steer if max_steer > 1e-6 else 0
        steering_vel_factor = max(0.1, np.cos(norm_steer_mag * np.pi / 2))

        min_dist_overall_eff = min_dist_to_static_boundary 
        for obs in perceived_dynamic_obstacles: 
             eff_dist = obs.shape.get_efficient_distance(robot_x, robot_y, *obs.get_position()) - self.robot_radius
             min_dist_overall_eff = min(min_dist_overall_eff, eff_dist)

        if min_dist_overall_eff < self.obstacle_avoid_distance:
            proximity_vel_factor = np.clip(min_dist_overall_eff / self.obstacle_avoid_distance, 0.0, 1.0)**1.5 
        elif min_dist_overall_eff < self.obstacle_slow_down_distance:
            slow_down_ratio = (min_dist_overall_eff - self.obstacle_avoid_distance) / (self.obstacle_slow_down_distance - self.obstacle_avoid_distance)
            proximity_vel_factor = 0.5 + 0.5 * slow_down_ratio 
            proximity_vel_factor = np.clip(proximity_vel_factor, 0.5, 1.0) 
        else:
            proximity_vel_factor = 1.0

        velocity = self.max_velocity * steering_vel_factor * proximity_vel_factor
        velocity = np.clip(velocity, self.min_velocity, self.max_velocity)

        action = np.array([velocity, steering_angle], dtype=np.float32)
        controller_info = self._get_controller_info()

        return action, controller_info

    # _get_lookahead_point method is REMOVED

    def _is_path_invalidated(self, robot_x, robot_y):
        # Path needs at least 2 points to form a segment.
        # If current_path_target_idx is the last point, no segments *from* it to check.
        if not self.current_planned_path or len(self.current_planned_path) < 2 or \
           self.current_path_target_idx >= len(self.current_planned_path) - 1:
             return False

        # Segments to check start from the current target waypoint.
        # e.g., if current_path_target_idx is k, we check segment (path[k] to path[k+1]), then (path[k+1] to path[k+2]), etc.
        start_check_idx = self.current_path_target_idx 
        
        # Max index for the first point of a segment is len(path) - 2 (for segment path[len-2]-path[len-1])
        # Horizon is number of segments, so end_idx for first point is start_idx + horizon - 1
        max_possible_start_idx_for_last_segment_check = len(self.current_planned_path) - 2
        end_check_idx = min(max_possible_start_idx_for_last_segment_check, 
                            start_check_idx + self.path_invalidation_check_horizon - 1)
        
        # Filter static obstacles for path invalidation check
        static_obstacles = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.STATIC]

        for i in range(start_check_idx, end_check_idx + 1):
             p1 = self.current_planned_path[i]
             p2 = self.current_planned_path[i+1]
             segment_valid = True # Assume valid until proven otherwise
             for obs in static_obstacles:
                  if obs.intersects_segment(p1, p2, self.robot_radius):
                      print(f"Path invalidated: Segment ({i})-({i+1}) from P[{i}] to P[{i+1}] blocked by perceived {type(obs.shape).__name__} near ({obs.x:.1f},{obs.y:.1f})")
                      segment_valid = False
                      break 
             if not segment_valid:
                  return True # Path is blocked ahead

        return False

    def _normalize_angle(self, angle):
        while angle > np.pi: angle -= 2 * np.pi
        while angle < -np.pi: angle += 2 * np.pi
        return angle

    def _slerp_angle(self, a1, a2, t):
        a1 = self._normalize_angle(a1); a2 = self._normalize_angle(a2)
        diff = self._normalize_angle(a2 - a1)
        return self._normalize_angle(a1 + diff * t)

    def _get_controller_info(self):
        return {
            "status": self.status,
            "planned_path": self.current_planned_path,
            "rrt_nodes": self.current_rrt_nodes,
            "rrt_parents": self.current_rrt_parents,
            "target_idx": self.current_path_target_idx
            }