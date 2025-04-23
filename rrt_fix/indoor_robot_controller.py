import time
import math
import numpy as np
from collections import namedtuple

from indoor_robot_env import IndoorRobotEnv
from components.shape import Shape, Circle, Rectangle
from components.obstacle import ObstacleType

from utils.obstacle_identifier import ObstacleIdentifier
from utils.ray_tracing_algorithm import RayTracingAlgorithm
from utils.rrt_planner import RRTPathPlanner
from utils.waiting_rule import WaitingRule

PlannerObstacleAdapter = namedtuple("PlannerObstacleAdapter", 
                                    ["x", "y", "shape", "velocity", "direction", "type", "is_dynamic"])

class IndoorRobotController:
    def __init__(self, env: IndoorRobotEnv):
        self.width = env.width
        self.height = env.height
        self.robot_radius = env.robot_radius
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.obstacle_identifier = ObstacleIdentifier(env.max_obstacles_in_observation)
        self.ray_tracer = RayTracingAlgorithm(self.width, self.height, self.robot_radius)
        self.waiting_rule = WaitingRule(self.robot_radius)
        self.path_planner = RRTPathPlanner(self.width, self.height, self.robot_radius)

        self.goal_threshold = self.robot_radius + 5
        self.max_velocity = self.action_space.high[0]
        self.min_velocity = 0.3
        self.obstacle_slow_down_distance = self.robot_radius * 5
        self.obstacle_avoid_distance = self.robot_radius * 3
        self.lookahead_distance = self.robot_radius * 4
        self.path_invalidation_check_horizon = 10
        self.collision_prediction_time = 3.0  # Increased for dynamic obstacles

        self.current_planned_path = None
        self.current_rrt_nodes = None
        self.current_rrt_parents = None
        self.current_path_target_idx = 0
        self.perceived_obstacles_data = []
        self.map_obstacles_used_for_plan = []
        self.is_waiting = False
        self.last_replanning_time = -np.inf
        self.status = "Initializing"
        self.debug_info = {}

    def reset(self):
        self.current_planned_path = None
        self.current_rrt_nodes = None
        self.current_rrt_parents = None
        self.current_path_target_idx = 0
        self.perceived_obstacles_data = []
        self.map_obstacles_used_for_plan = []
        self.is_waiting = False
        self.last_replanning_time = -np.inf
        self.status = "Reset"
        self.debug_info = {}

    def get_action(self, observation: np.ndarray):
        current_time = time.time()
        self.status = "Processing"
        self.debug_info = {}

        robot_x, robot_y, robot_orientation, goal_x, goal_y = observation[:IndoorRobotEnv.OBS_ROBOT_STATE_SIZE]
        self.perceived_obstacles_data = self.obstacle_identifier.identify(observation)
        self.debug_info["obstacles_count"] = len(self.perceived_obstacles_data)

        for obs_data in self.perceived_obstacles_data:
            if not obs_data['is_dynamic'] and np.linalg.norm(obs_data['direction'] * obs_data['velocity']) > 0.1:
                print(f"Warning: Obstacle @({obs_data['x']:.1f},{obs_data['y']:.1f}) marked static but has velocity ({obs_data['velocity']:.1f}). Correcting to dynamic.")
                obs_data['is_dynamic'] = True

        distance_to_goal = np.linalg.norm(np.array([robot_x, robot_y]) - np.array([goal_x, goal_y]))
        if distance_to_goal < self.goal_threshold:
            self.status = "Goal Reached - Stopping"
            self.current_planned_path = None
            self.current_path_target_idx = 0
            info = self._get_controller_info()
            info['goal_reached'] = True
            return np.array([0.0, 0.0]), info

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
            self.debug_info["path_invalidated"] = True

        if needs_replan:
            self.status += " - Replanning..."
            print(self.status)

            static_obstacles_data = [obs for obs in self.perceived_obstacles_data if not obs['is_dynamic']]
            if not static_obstacles_data:
                print("Warning: No static obstacles detected for RRT planning. Path may be trivial.")

            planner_map_objects = []
            for obs_data in static_obstacles_data:
                adapter = PlannerObstacleAdapter(
                    x=obs_data['x'], 
                    y=obs_data['y'], 
                    shape=obs_data['shape'],
                    velocity=0.0, 
                    direction=np.array([0.0, 0.0]), 
                    type=ObstacleType.STATIC,
                    is_dynamic=False
                )
                planner_map_objects.append(adapter)
            
            self.map_obstacles_used_for_plan = planner_map_objects
            self.debug_info["static_obstacles_for_planning"] = len(planner_map_objects)

            new_path, nodes, parents = self.path_planner.plan_path(
                robot_x, robot_y, goal_x, goal_y, planner_map_objects
            )

            if new_path and len(new_path) > 1:
                self.current_planned_path = new_path
                self.current_rrt_nodes = nodes
                self.current_rrt_parents = parents
                self.current_path_target_idx = 0
                self.last_replanning_time = current_time
                self.status = "Replanning Successful"
                self.debug_info["replanning"] = "success"
            else:
                self.status = "Replanning Failed"
                print(self.status)
                self.current_planned_path = None
                self.current_path_target_idx = 0
                self.current_rrt_nodes = nodes
                self.current_rrt_parents = parents
                self.debug_info["replanning"] = "failed"
                return np.array([0.0, 0.0]), self._get_controller_info()

        if self.current_planned_path is None or len(self.current_planned_path) <= 1:
            self.status = "No valid path - Stopping"
            return np.array([0.0, 0.0]), self._get_controller_info()

        perceived_dynamic_obstacles_data = [obs for obs in self.perceived_obstacles_data if obs['is_dynamic']]
        potential_velocity = self.max_velocity

        dynamic_adapter_objects = []
        for obs_data in perceived_dynamic_obstacles_data:
            adapter = PlannerObstacleAdapter(
                x=obs_data['x'], 
                y=obs_data['y'], 
                shape=obs_data['shape'],
                velocity=obs_data['velocity'], 
                direction=obs_data['direction'],
                type=ObstacleType.DYNAMIC,
                is_dynamic=True
            )
            dynamic_adapter_objects.append(adapter)
        
        self.debug_info["dynamic_obstacles"] = len(dynamic_adapter_objects)

        predicted_collisions = self.waiting_rule.check_dynamic_collisions(
            robot_x, robot_y, potential_velocity, robot_orientation, dynamic_adapter_objects
        )

        if self.waiting_rule.should_wait(predicted_collisions):
            colliding_obs_info = []
            if isinstance(predicted_collisions, list):
                colliding_obs_info = [
                    (f"Obs({type(obs_adapter.shape).__name__})", f"{t:.1f}s")
                    for obs_adapter, t in predicted_collisions
                ]
            self.status = f"Waiting for dynamic obstacle(s): {colliding_obs_info}"
            self.is_waiting = True
            self.debug_info["waiting"] = True
            return np.array([0.0, 0.0]), self._get_controller_info()
        else:
            self.is_waiting = False

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
            if self.current_path_target_idx < len(self.current_planned_path) - 1:
                next_target = self.current_planned_path[self.current_path_target_idx + 1]
                target_vector = np.array([next_target[0] - robot_x, next_target[1] - robot_y])
                target_orientation = np.arctan2(target_vector[1], target_vector[0]) if np.linalg.norm(target_vector) > 1e-6 else robot_orientation
            else:
                target_orientation = robot_orientation
        else:
            target_orientation = np.arctan2(target_vector[1], target_vector[0])

        final_target_orientation = target_orientation
        min_dist_to_static_boundary = float('inf')
        avoiding_obstacle_data = None
        current_min_avoid_eff_dist = float('inf')

        perceived_static_obstacles_data = [obs for obs in self.perceived_obstacles_data if not obs['is_dynamic']]

        for obs_data in perceived_static_obstacles_data:
            eff_dist = obs_data['shape'].get_efficient_distance(
                robot_x, robot_y, obs_data['x'], obs_data['y']
            ) - self.robot_radius
            min_dist_to_static_boundary = min(min_dist_to_static_boundary, eff_dist)

            if eff_dist < self.obstacle_avoid_distance:
                vec_to_obs = np.array([obs_data['x'], obs_data['y']]) - np.array([robot_x, robot_y])
                angle_to_obs = np.arctan2(vec_to_obs[1], vec_to_obs[0])
                angle_diff_to_target = abs(self._normalize_angle(angle_to_obs - target_orientation))
                if angle_diff_to_target < np.pi / 1.9:
                    if avoiding_obstacle_data is None or eff_dist < current_min_avoid_eff_dist:
                        avoiding_obstacle_data = obs_data
                        current_min_avoid_eff_dist = eff_dist

        if avoiding_obstacle_data is not None:
            final_avoid_eff_dist = avoiding_obstacle_data['shape'].get_efficient_distance(
                robot_x, robot_y, avoiding_obstacle_data['x'], avoiding_obstacle_data['y']
            ) - self.robot_radius
            self.status = f"Avoiding static {type(avoiding_obstacle_data['shape']).__name__}"
            self.debug_info["avoiding"] = type(avoiding_obstacle_data['shape']).__name__

            obstacle_adapter = PlannerObstacleAdapter(
                x=avoiding_obstacle_data['x'],
                y=avoiding_obstacle_data['y'],
                shape=avoiding_obstacle_data['shape'],
                velocity=0.0,
                direction=np.array([0.0, 0.0]),
                type=ObstacleType.STATIC,
                is_dynamic=False
            )

            avoidance_orientation = self.ray_tracer.avoid_static_obstacle(
                robot_x, robot_y, robot_orientation, obstacle_adapter, goal_x, goal_y
            )

            blend_factor = 1.0 - np.clip(final_avoid_eff_dist / self.obstacle_avoid_distance, 0.0, 1.0)
            blend_factor = blend_factor**2
            final_target_orientation = self._slerp_angle(target_orientation, avoidance_orientation, blend_factor)

        steering_angle = self._normalize_angle(final_target_orientation - robot_orientation)
        steering_angle = np.clip(steering_angle, self.action_space.low[1], self.action_space.high[1])

        max_steer = abs(self.action_space.high[1])
        norm_steer = abs(steering_angle) / max_steer if max_steer > 1e-6 else 0
        steering_vel_factor = max(0.1, np.cos(norm_steer * np.pi / 2))

        min_dist_overall_eff = min_dist_to_static_boundary
        for obs_data in perceived_dynamic_obstacles_data:
            eff_dist = obs_data['shape'].get_efficient_distance(
                robot_x, robot_y, obs_data['x'], obs_data['y']
            ) - self.robot_radius
            min_dist_overall_eff = min(min_dist_overall_eff, eff_dist)

        if min_dist_overall_eff < self.obstacle_avoid_distance:
            proximity_vel_factor = np.clip(min_dist_overall_eff / self.obstacle_avoid_distance, 0.0, 1.0)**1.5
        elif min_dist_overall_eff < self.obstacle_slow_down_distance:
            slow_down_ratio = (min_dist_overall_eff - self.obstacle_avoid_distance) / (
                self.obstacle_slow_down_distance - self.obstacle_avoid_distance)
            proximity_vel_factor = 0.5 + 0.5 * slow_down_ratio
            proximity_vel_factor = np.clip(proximity_vel_factor, 0.5, 1.0)
        else:
            proximity_vel_factor = 1.0

        velocity = self.max_velocity * steering_vel_factor * proximity_vel_factor
        velocity = np.clip(velocity, self.min_velocity, self.max_velocity)

        action = np.array([velocity, steering_angle], dtype=np.float32)
        controller_info = self._get_controller_info()

        return action, controller_info

    def _get_lookahead_point(self, robot_x, robot_y):
        if not self.current_planned_path or len(self.current_planned_path) < 2:
            return None, self.current_path_target_idx

        robot_pos = np.array([robot_x, robot_y])
        current_target_idx = self.current_path_target_idx

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
                projection_t = t

        if closest_segment_idx >= current_target_idx:
            current_target_idx = min(closest_segment_idx + 1, len(self.current_planned_path) - 1)
        elif closest_segment_idx == current_target_idx - 1 and projection_t > 1.0:
            current_target_idx = min(current_target_idx + 1, len(self.current_planned_path) - 1)

        lookahead_point_found = None
        start_node_idx = max(0, current_target_idx - 1)
        start_search_point_on_path = np.array(self.current_planned_path[start_node_idx])
        if start_node_idx < len(self.current_planned_path) - 1:
            p1 = np.array(self.current_planned_path[start_node_idx])
            p2 = np.array(self.current_planned_path[start_node_idx+1])
            seg_vec = p2 - p1
            seg_len_sq = np.dot(seg_vec, seg_vec)
            if seg_len_sq > 1e-9:
                t = np.dot(robot_pos - p1, seg_vec) / seg_len_sq
                t_clamped = np.clip(t, 0, 1)
                start_search_point_on_path = p1 + t_clamped * seg_vec

        current_dist_on_path = 0.0
        for i in range(start_node_idx, len(self.current_planned_path) - 1):
            p1 = np.array(self.current_planned_path[i])
            p2 = np.array(self.current_planned_path[i+1])
            segment_vec = p2 - p1
            segment_len = np.linalg.norm(segment_vec)
            if segment_len < 1e-6:
                continue
            dist_to_p1 = np.linalg.norm(start_search_point_on_path - p1)
            dist_to_p2 = np.linalg.norm(start_search_point_on_path - p2)
            if i == start_node_idx and dist_to_p1 < segment_len and dist_to_p2 < segment_len:
                dist_remaining_on_segment = segment_len - dist_to_p1
            else:
                dist_remaining_on_segment = segment_len
            if current_dist_on_path + dist_remaining_on_segment >= self.lookahead_distance:
                needed_dist = self.lookahead_distance - current_dist_on_path
                dist_on_this_segment = segment_len - dist_remaining_on_segment + needed_dist
                ratio = np.clip(dist_on_this_segment / segment_len, 0, 1)
                lookahead_point_found = tuple(p1 + ratio * segment_vec)
                break
            else:
                current_dist_on_path += dist_remaining_on_segment
                start_search_point_on_path = p2

        if lookahead_point_found is None:
            lookahead_point_found = self.current_planned_path[-1]

        return lookahead_point_found, current_target_idx

    def _is_path_invalidated(self, robot_x, robot_y):
        if not self.current_planned_path or len(self.current_planned_path) < 2:
            return False
        if self.current_path_target_idx >= len(self.current_planned_path) - 1:
            return False

        robot_pos = np.array([robot_x, robot_y])
        min_dist_sq = float('inf')
        closest_segment_idx = self.current_path_target_idx
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
                closest_point = p1 + t * seg_vec
                dist_sq = np.dot(robot_pos - closest_point, robot_pos - closest_point)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_segment_idx = i

        start_check_idx = closest_segment_idx
        end_check_idx = min(len(self.current_planned_path) - 2, start_check_idx + self.path_invalidation_check_horizon)

        for i in range(start_check_idx, end_check_idx + 1):
            p1 = self.current_planned_path[i]
            p2 = self.current_planned_path[i+1]
            for obs_data in self.perceived_obstacles_data:
                shape_obj = obs_data['shape']
                obs_x, obs_y = obs_data['x'], obs_data['y']
                dist_to_obs = np.sqrt((obs_x - robot_x)**2 + (obs_y - robot_y)**2)
                print(f"Checking obstacle @({obs_x:.1f},{obs_y:.1f}), dist={dist_to_obs:.1f}, "
                      f"Type={'Dynamic' if obs_data['is_dynamic'] else 'Static'}")
                # Check current position
                if shape_obj.intersects_segment(p1, p2, self.robot_radius * 1.5, obs_x, obs_y):
                    shape_name = type(shape_obj).__name__
                    print(f"Path invalidated: Segment {i} blocked by {shape_name} at ({obs_x:.1f},{obs_y:.1f})")
                    self.debug_info["invalidation_segment"] = i
                    self.debug_info["invalidation_obstacle"] = f"{shape_name}@({obs_x:.1f},{obs_y:.1f})"
                    return True
                # Check projected position for dynamic obstacles
                if obs_data['is_dynamic']:
                    vel = obs_data['velocity'] * obs_data['direction']
                    for t in np.linspace(0, self.collision_prediction_time, 10):  # More points
                        proj_x = obs_x + vel[0] * t
                        proj_y = obs_y + vel[1] * t
                        if shape_obj.intersects_segment(p1, p2, self.robot_radius * 1.5, proj_x, proj_y):
                            shape_name = type(shape_obj).__name__
                            print(f"Path invalidated: Segment {i} blocked by projected {shape_name} at ({proj_x:.1f},{proj_y:.1f}) in {t:.1f}s")
                            self.debug_info["invalidation_segment"] = i
                            self.debug_info["invalidation_obstacle"] = f"{shape_name}@({proj_x:.1f},{proj_y:.1f})"
                            return True
                        # Check proximity for dynamic obstacles
                        segment_mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                        dist_to_segment = np.sqrt((proj_x - segment_mid[0])**2 + (proj_y - segment_mid[1])**2)
                        if dist_to_segment < self.robot_radius * 3:
                            print(f"Path invalidated: Dynamic obstacle @({proj_x:.1f},{proj_y:.1f}) too close to segment {i} (dist={dist_to_segment:.1f})")
                            self.debug_info["invalidation_segment"] = i
                            self.debug_info["invalidation_obstacle"] = f"Dynamic@{proj_x:.1f},{proj_y:.1f}"
                            return True
        print(f"No path invalidation for segments {start_check_idx} to {end_check_idx}")
        return False

    def _normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _slerp_angle(self, a1, a2, t):
        a1 = self._normalize_angle(a1)
        a2 = self._normalize_angle(a2)
        diff = self._normalize_angle(a2 - a1)
        return self._normalize_angle(a1 + diff * t)

    def _get_controller_info(self):
        return {
            "status": self.status,
            "planned_path": self.current_planned_path,
            "rrt_nodes": self.current_rrt_nodes,
            "rrt_parents": self.current_rrt_parents,
            "target_idx": self.current_path_target_idx,
            "debug_info": self.debug_info
        }