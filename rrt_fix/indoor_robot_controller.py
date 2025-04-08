import time
import math
import numpy as np
from indoor_robot_env import IndoorRobotEnv
from obstacle import Obstacle, ObstacleType
from shape import Circle, Rectangle
from utils.ray_tracing_algorithm import RayTracingAlgorithm
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
        self.ray_tracer = RayTracingAlgorithm(self.width, self.height, self.robot_radius) # Ray tracer less useful without proper intersection logic
        self.waiting_rule = WaitingRule(self.robot_radius)
        self.path_planner = RRTPathPlanner(self.width, self.height, self.robot_radius)

        # Controller parameters (mostly same)
        self.goal_threshold = self.robot_radius + 5
        self.max_velocity = self.action_space.high[0]
        self.min_velocity = 0.2
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

    def get_action(self, observation: np.ndarray):
        current_time = time.time()
        self.status = "Processing"

        # 1. Perceive Environment - Uses new ObstacleIdentifier
        robot_x, robot_y, robot_orientation, goal_x, goal_y = observation[:IndoorRobotEnv.OBS_ROBOT_STATE_SIZE]
        self.perceived_obstacles = self.obstacle_identifier.identify(observation)
        # print(f"Perceived {len(self.perceived_obstacles)} obstacles.")

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
        if self.current_planned_path is None or len(self.current_planned_path) <= 1:
             self.status = "No valid path"
             needs_replan = True
        elif self.current_path_target_idx >= len(self.current_planned_path):
             self.status = "Target index out of bounds"
             needs_replan = True
             self.current_planned_path = None
        elif self._is_path_invalidated(robot_x, robot_y):
             self.status = "Path invalidated by new obstacle"
             needs_replan = True

        if needs_replan:
            self.status += " - Replanning..."
            print(self.status)
            # Update the "map" for the planner.
            # CRITICAL: Reset dynamic obstacles in the map to their initial positions
            # Or only include static obstacles in the map. Let's use static only for simplicity.
            # self.map_obstacles = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.STATIC]

            # Alternative: Use initial state of all *perceived* obstacles for planning
            # This requires storing initial state in perceived obstacles or copying from ground truth
            # Let's stick to using only currently perceived STATIC obstacles for the map.
            current_static_map = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.STATIC]
            # Need a deep copy if planner modifies the list/objects (RRT doesn't modify them)
            self.map_obstacles = current_static_map # Store for info

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
                 print(self.status)
                 self.current_planned_path = None
                 self.current_path_target_idx = 0
                 self.current_rrt_nodes = nodes # Store tree for viz
                 self.current_rrt_parents = parents
                 return np.array([0.0, 0.0]), self._get_controller_info() # Stop if no path

        # If still no valid path after trying to replan, stop
        if self.current_planned_path is None or len(self.current_planned_path) <= 1:
             self.status = "No valid path - Stopping"
             return np.array([0.0, 0.0]), self._get_controller_info()


        # 4. Reactive Behavior - Dynamic Obstacles (Waiting Rule)
        perceived_dynamic_obstacles = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.DYNAMIC]
        # Use potential velocity for prediction
        potential_velocity = self.max_velocity # Or velocity from previous step? Let's use max.

        predicted_collisions = self.waiting_rule.check_dynamic_collisions(
            robot_x, robot_y, potential_velocity, robot_orientation, perceived_dynamic_obstacles
        )

        if self.waiting_rule.should_wait(predicted_collisions):
             colliding_obs_info = [(f"ObsType({type(obs.shape).__name__})", t) for obs, t in predicted_collisions]
             self.status = f"Waiting for dynamic obstacle(s): {colliding_obs_info}"
             self.is_waiting = True
             return np.array([0.0, 0.0]), self._get_controller_info() # Stop action
        else:
             self.is_waiting = False


        # 5. Path Following & Static Obstacle Avoidance
        self.status = "Following Path"
        lookahead_point, self.current_path_target_idx = self._get_lookahead_point(robot_x, robot_y)

        if lookahead_point is None:
             self.status = "Path Error (Lookahead) - Stopping"
             print(self.status)
             self.current_planned_path = None
             return np.array([0.0, 0.0]), self._get_controller_info()

        target_x, target_y = lookahead_point
        target_vector = np.array([target_x - robot_x, target_y - robot_y])
        target_distance = np.linalg.norm(target_vector)

        if target_distance < 1e-6:
             # Aim for next point or maintain heading
             if self.current_path_target_idx < len(self.current_planned_path) - 1:
                  next_target = self.current_planned_path[self.current_path_target_idx + 1]
                  target_vector = np.array([next_target[0] - robot_x, next_target[1] - robot_y])
                  target_orientation = np.arctan2(target_vector[1], target_vector[0]) if np.linalg.norm(target_vector) > 1e-6 else robot_orientation
             else:
                  target_orientation = robot_orientation
        else:
             target_orientation = np.arctan2(target_vector[1], target_vector[0])

        # --- Reactive Static Obstacle Avoidance ---
        final_target_orientation = target_orientation
        min_dist_to_static_boundary = float('inf')
        avoiding_obstacle = None

        perceived_static_obstacles = [obs for obs in self.perceived_obstacles if obs.type == ObstacleType.STATIC]
        for obstacle in perceived_static_obstacles:
             # Calculate distance to obstacle boundary (approximate needed here)
             # Use check_collision logic: find closest point on shape to robot center
             # This requires exposing that logic or re-implementing here.
             # Simplification: Use distance to center minus estimated radius.
             dist_center = np.linalg.norm(np.array([robot_x, robot_y]) - obstacle.get_position())
             est_radius = 0
             if isinstance(obstacle.shape, Circle): est_radius = obstacle.shape.radius
             elif isinstance(obstacle.shape, Rectangle): est_radius = 0.5 * math.sqrt(obstacle.shape.width**2 + obstacle.shape.height**2)
             eff_dist = dist_center - est_radius - self.robot_radius # Approx dist from robot boundary to obs boundary

             min_dist_to_static_boundary = min(min_dist_to_static_boundary, eff_dist)

             if eff_dist < self.obstacle_avoid_distance:
                  vec_to_obs = obstacle.get_position() - np.array([robot_x, robot_y])
                  angle_to_obs = np.arctan2(vec_to_obs[1], vec_to_obs[0])
                  angle_diff_to_target = abs(self._normalize_angle(angle_to_obs - target_orientation))

                  # Avoid if somewhat in front
                  if angle_diff_to_target < np.pi / 1.9:
                      # Use the closest one in the avoidance cone
                      if avoiding_obstacle is None or eff_dist < min_dist_to_static_boundary: # Bug: should compare eff_dist here
                          # Corrected logic: track min effective distance *for avoidance candidates*
                          current_min_avoid_eff_dist = float('inf')
                          if avoiding_obstacle: # If already avoiding one, get its eff_dist again
                              # Recalculate eff_dist for the current 'avoiding_obstacle' to compare
                              avoid_dist_center = np.linalg.norm(np.array([robot_x, robot_y]) - avoiding_obstacle.get_position())
                              avoid_est_radius = 0
                              if isinstance(avoiding_obstacle.shape, Circle): avoid_est_radius = avoiding_obstacle.shape.radius
                              elif isinstance(avoiding_obstacle.shape, Rectangle): avoid_est_radius = 0.5 * math.sqrt(avoiding_obstacle.shape.width**2 + avoiding_obstacle.shape.height**2)
                              current_min_avoid_eff_dist = avoid_dist_center - avoid_est_radius - self.robot_radius

                          if eff_dist < current_min_avoid_eff_dist:
                               avoiding_obstacle = obstacle


        if avoiding_obstacle is not None:
            # Recalculate final eff_dist for the chosen obstacle for blending
            avoid_dist_center = np.linalg.norm(np.array([robot_x, robot_y]) - avoiding_obstacle.get_position())
            avoid_est_radius = 0
            if isinstance(avoiding_obstacle.shape, Circle): avoid_est_radius = avoiding_obstacle.shape.radius
            elif isinstance(avoiding_obstacle.shape, Rectangle): avoid_est_radius = 0.5 * math.sqrt(avoiding_obstacle.shape.width**2 + avoiding_obstacle.shape.height**2)
            final_avoid_eff_dist = avoid_dist_center - avoid_est_radius - self.robot_radius

            self.status = f"Avoiding static {type(avoiding_obstacle.shape).__name__}"
            avoidance_orientation = self.ray_tracer.avoid_static_obstacle( # Uses simplified centroid-based logic
                robot_x, robot_y, robot_orientation, avoiding_obstacle, goal_x, goal_y
            )
            # Blend based on proximity
            blend_factor = 1.0 - np.clip(final_avoid_eff_dist / self.obstacle_avoid_distance, 0.0, 1.0)
            blend_factor = blend_factor**2
            final_target_orientation = self._slerp_angle(target_orientation, avoidance_orientation, blend_factor)


        # 6. Calculate Control Action
        steering_angle = self._normalize_angle(final_target_orientation - robot_orientation)
        steering_angle = np.clip(steering_angle, self.action_space.low[1], self.action_space.high[1])

        velocity = self.max_velocity
        # Velocity scaling based on steering
        max_steer = abs(self.action_space.high[1])
        norm_steer_mag = abs(steering_angle) / max_steer if max_steer > 1e-6 else 0
        steering_vel_factor = max(0.1, np.cos(norm_steer_mag * np.pi / 2))

        # Velocity scaling based on proximity to *any* obstacle boundary
        min_dist_overall_eff = min_dist_to_static_boundary # Start with closest static
        for obs in perceived_dynamic_obstacles: # Also consider dynamic ones
             dist_center = np.linalg.norm(np.array([robot_x, robot_y]) - obs.get_position())
             est_radius = 0
             if isinstance(obs.shape, Circle): est_radius = obs.shape.radius
             elif isinstance(obs.shape, Rectangle): est_radius = 0.5 * math.sqrt(obs.shape.width**2 + obs.shape.height**2)
             eff_dist = dist_center - est_radius - self.robot_radius
             min_dist_overall_eff = min(min_dist_overall_eff, eff_dist)

        proximity_vel_factor = np.clip(min_dist_overall_eff / self.obstacle_slow_down_distance, 0.0, 1.0)

        velocity = self.max_velocity * steering_vel_factor * proximity_vel_factor
        velocity = np.clip(velocity, self.min_velocity, self.max_velocity)

        action = np.array([velocity, steering_angle], dtype=np.float32)
        controller_info = self._get_controller_info()

        return action, controller_info


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

        for i in range(search_start_idx, len(self.current_planned_path) - 1):
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
             current_target_idx = min(current_target_idx + 1, len(self.current_planned_path) - 1)

        # --- Find Lookahead Point ---
        lookahead_point_found = None
        dist_along_path = 0.0
        start_node_idx = max(0, current_target_idx - 1)
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
            p1 = np.array(self.current_planned_path[i])
            p2 = np.array(self.current_planned_path[i+1])
            segment_vec = p2 - p1
            segment_len = np.linalg.norm(segment_vec)

            if segment_len < 1e-6: continue

            # How much of this segment is relevant for lookahead search?
            dist_to_p1 = np.linalg.norm(start_search_point_on_path - p1)
            dist_to_p2 = np.linalg.norm(start_search_point_on_path - p2)

            if i == start_node_idx and dist_to_p1 < segment_len and dist_to_p2 < segment_len: # Robot projection is on this segment
                 dist_remaining_on_segment = segment_len - dist_to_p1
            else: # Robot is before p1 or on a previous segment
                 dist_remaining_on_segment = segment_len

            if current_dist_on_path + dist_remaining_on_segment >= self.lookahead_distance:
                # Lookahead point is on this segment
                needed_dist = self.lookahead_distance - current_dist_on_path
                # Calculate the fraction needed along the *full* segment length
                # Start from p1 and move 'dist_on_this_segment'
                dist_on_this_segment = segment_len - dist_remaining_on_segment + needed_dist # Dist from p1
                ratio = dist_on_this_segment / segment_len
                ratio = np.clip(ratio, 0, 1)
                lookahead_point_found = tuple(p1 + ratio * segment_vec)
                break
            else:
                current_dist_on_path += dist_remaining_on_segment
                start_search_point_on_path = p2 # Advance the reference point

        if lookahead_point_found is None:
            lookahead_point_found = self.current_planned_path[-1] # Use last point if path too short

        return lookahead_point_found, current_target_idx


    def _is_path_invalidated(self, robot_x, robot_y):
        # Uses the planner's segment check which now handles Obstacle objects
        if not self.current_planned_path or len(self.current_planned_path) < 2 or self.current_path_target_idx >= len(self.current_planned_path) - 1:
             return False

        # Find closest segment index to robot (reuse logic from lookahead)
        robot_pos = np.array([robot_x, robot_y])
        min_dist_sq = float('inf')
        closest_segment_idx = self.current_path_target_idx
        search_start_idx = max(0, self.current_path_target_idx - 1)
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
             p1 = self.current_planned_path[i]
             p2 = self.current_planned_path[i+1]
             # Check this path segment against *currently perceived obstacles* using planner's utility
             # Needs access to RRT's collision checker or reimplement here.
             # Reimplement using obstacle.intersects_segment:
             segment_valid = True
             for obs in self.perceived_obstacles:
                  # Check segment against perceived obstacle shape
                  if obs.intersects_segment(p1, p2, self.robot_radius):
                      print(f"Path invalidated: Segment {i} blocked by perceived {type(obs.shape).__name__} near ({obs.x:.1f},{obs.y:.1f})")
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
            # "rays": self.last_ray_viz_points (if rays implemented fully)
            }