# --- START OF FILE ray_tracing_algorithm.py ---

import numpy as np
import math
from components.obstacle import Obstacle
from components.shape import Shape # Import base Shape class

class RayTracingAlgorithm:
    def __init__(self, env_width, env_height, robot_radius, sensor_range):
        self.width = env_width
        self.height = env_height
        self.robot_radius = robot_radius # Store robot radius for padding/safety checks
        self.sensor_range = sensor_range

        # Parameters for the ray-based avoidance
        self.num_avoidance_rays = 21 # Number of rays to cast for finding gaps
        self.avoidance_ray_angle_span = np.pi * 0.8 # Angular range to cast rays (e.g., 144 degrees)
        self.avoidance_ray_max_dist = self.sensor_range # Max distance to check for intersections relevant to avoidance
        self.safety_distance_ratio = 0 # Multiplier for robot radius to ensure clearance

    def _normalize_angle(self, angle):
        """ Normalizes angle to be within [-pi, pi]. """
        while angle > np.pi: angle -= 2 * np.pi
        while angle < -np.pi: angle += 2 * np.pi
        return angle

    # --- trace_rays (Kept for general purpose/visualization, but not directly used by avoid_static_obstacle) ---
    def trace_rays(self, robot_x, robot_y, robot_orientation, obstacles: list, num_rays=36, max_ray_length=150):
        """
        Traces multiple rays outwards and finds the first intersection with any obstacle
        or boundary using the Shape.intersect_ray method.

        Args:
            robot_x, robot_y: Robot position.
            robot_orientation: Robot orientation (used as center for ray spread).
            obstacles (list): List of Obstacle objects to check against.
            num_rays (int): Number of rays to cast (evenly spaced 2pi).
            max_ray_length (float): Maximum length for rays.

        Returns:
            tuple: (ray_intersections, ray_viz_points)
                ray_intersections: List of tuples (intersect_x, intersect_y, intersected_obstacle_or_boundary_flag).
                ray_viz_points: List of tuples ((start_x, start_y), (end_x, end_y)) for visualization.
        """
        ray_intersections = []
        ray_viz_points = []
        robot_pos = np.array([robot_x, robot_y])

        for i in range(num_rays):
            # Cast rays evenly around 2*pi relative to world frame
            angle = self._normalize_angle(i * (2 * np.pi / num_rays))
            ray_dir = np.array([np.cos(angle), np.sin(angle)])

            closest_t = max_ray_length
            intersected_object = None # None initially

            # 1. Check intersection with provided obstacles
            for obs in obstacles:
                # Ensure the Shape class has the intersect_ray method implemented
                if hasattr(obs.shape, 'intersect_ray') and callable(obs.shape.intersect_ray):
                    # Pass obstacle's current position (obs.x, obs.y)
                    t = obs.shape.intersect_ray(robot_pos, ray_dir, obs.x, obs.y)
                    # Add a safety margin based on robot radius when comparing distances
                    # If intersection distance is less than closest found so far (considering safety margin)
                    effective_t = t - self.robot_radius * self.safety_distance_ratio
                    if effective_t < closest_t and t >= 0: # Check t>=0 to avoid hits behind the origin
                        closest_t = max(0, effective_t) # Use effective distance, ensure non-negative
                        intersected_object = obs # Store the obstacle object
                else:
                     # Fallback or warning if method is missing
                     # print(f"Warning: Shape {type(obs.shape)} missing intersect_ray method.")
                     pass # Or use a bounding box approximation? For now, skip.


            # 2. Check intersection with environment boundaries (optional, can be simplified)
            boundary_t = max_ray_length
            # Left (x=0)
            if abs(ray_dir[0]) > 1e-6 and ray_dir[0] < 0:
                t_bound = -robot_x / ray_dir[0]
                if 0 <= t_bound < boundary_t: boundary_t = t_bound
            # Right (x=width)
            if abs(ray_dir[0]) > 1e-6 and ray_dir[0] > 0:
                t_bound = (self.width - robot_x) / ray_dir[0]
                if 0 <= t_bound < boundary_t: boundary_t = t_bound
            # Bottom (y=0)
            if abs(ray_dir[1]) > 1e-6 and ray_dir[1] < 0:
                t_bound = -robot_y / ray_dir[1]
                if 0 <= t_bound < boundary_t: boundary_t = t_bound
            # Top (y=height)
            if abs(ray_dir[1]) > 1e-6 and ray_dir[1] > 0:
                t_bound = (self.height - robot_y) / ray_dir[1]
                if 0 <= t_bound < boundary_t: boundary_t = t_bound

            # If boundary is closer than obstacle intersection (considering safety)
            if boundary_t < closest_t:
                closest_t = boundary_t
                intersected_object = "Boundary" # Flag for boundary hit

            # Calculate final intersection point based on the minimum t found
            closest_t = min(closest_t, max_ray_length) # Ensure it doesn't exceed max length
            intersection_point = robot_pos + ray_dir * closest_t

            ray_intersections.append((*intersection_point, intersected_object))
            ray_viz_points.append(((robot_x, robot_y), tuple(intersection_point)))

        return ray_intersections, ray_viz_points


    # --- avoid_static_obstacle (Revised Implementation using Ray Casting Idea) ---
    def avoid_static_obstacle(self, robot_x, robot_y, robot_radius, robot_orientation, obstacle: Obstacle, lookahead_x, lookahead_y):
        """
        Calculates an avoidance orientation by casting rays to find gaps around a specific static obstacle.
        This attempts to mimic the outcome of the paper's ray tracing avoidance (Fig. 6)
        by finding a clear path around the *given* obstacle, prioritizing the direction
        towards the lookahead point.

        Args:
            robot_x, robot_y: Current robot position.
            robot_radius: Robot's radius (passed for consistency, uses self.robot_radius internally).
            robot_orientation: Current robot orientation (fallback).
            obstacle: The specific static Obstacle instance causing the avoidance check.
            lookahead_x, lookahead_y: The target point on the planned path.

        Returns:
            The calculated avoidance orientation (angle in radians).
        """
        robot_pos = np.array([robot_x, robot_y])
        obs_x, obs_y = obstacle.x, obstacle.y # Use obstacle's current position

        # --- Emergency Handling: Check for Overlap First ---
        # Use efficient distance check first
        dist_to_boundary = obstacle.shape.get_efficient_distance(robot_x, robot_y, obs_x, obs_y)

        # If robot is inside or touching (considering its radius)
        if dist_to_boundary <= self.robot_radius: # Use internal robot radius
            # Get vector pointing away from the closest point on the boundary
            vec_boundary_to_robot = obstacle.shape.get_effective_vector(robot_x, robot_y, obs_x, obs_y)
            dist_mag = np.linalg.norm(vec_boundary_to_robot)

            if dist_mag < 1e-6:
                 # Deep overlap or at center - move away from obstacle center
                 vec_center_to_robot = robot_pos - np.array([obs_x, obs_y])
                 if np.linalg.norm(vec_center_to_robot) < 1e-6:
                     # At the exact center? Turn around
                     # print("Avoidance: EMERGENCY (At Center) - Turning around")
                     return self._normalize_angle(robot_orientation + np.pi)
                 else:
                     angle_away_center = np.arctan2(vec_center_to_robot[1], vec_center_to_robot[0])
                     # print(f"Avoidance: EMERGENCY (Dist~0) - Moving away from center {angle_away_center:.2f}")
                     return self._normalize_angle(angle_away_center)
            else:
                 # Normal emergency: move directly away from the closest boundary point
                 angle_away = np.arctan2(vec_boundary_to_robot[1], vec_boundary_to_robot[0])
                 # print(f"Avoidance: EMERGENCY (Overlap/Touch Dist {dist_to_boundary:.2f}) - Moving away {angle_away:.2f}")
                 return self._normalize_angle(angle_away)

        # --- Ray Casting for Gaps ---
        # Calculate the desired direction towards the lookahead point
        vec_to_goal = np.array([lookahead_x - robot_x, lookahead_y - robot_y])
        dist_to_goal = np.linalg.norm(vec_to_goal)
        if dist_to_goal < 1e-6:
             angle_to_goal = robot_orientation # Default to current if already at goal
        else:
             angle_to_goal = np.arctan2(vec_to_goal[1], vec_to_goal[0])

        # Cast rays centered around the angle_to_goal
        ray_angles = np.linspace(
            angle_to_goal - self.avoidance_ray_angle_span / 2,
            angle_to_goal + self.avoidance_ray_angle_span / 2,
            self.num_avoidance_rays
        )

        ray_results = [] # Store (angle, intersection_distance)

        # Check if the shape has the required intersection method
        if not (hasattr(obstacle.shape, 'intersect_ray') and callable(obstacle.shape.intersect_ray)):
            print(f"Error: Obstacle shape {type(obstacle.shape)} lacks intersect_ray method. Cannot perform ray-based avoidance.")
            # Fallback: Use the old tangent-based method? Or just return goal angle?
            # Let's return goal angle as a simple fallback, assuming RRT path is clear enough
            return self._normalize_angle(angle_to_goal)


        for angle in ray_angles:
            norm_angle = self._normalize_angle(angle)
            ray_dir = np.array([np.cos(norm_angle), np.sin(norm_angle)])

            # Check intersection with the *specific obstacle*
            # Pass obstacle's current position (obs_x, obs_y)
            intersect_t = obstacle.shape.intersect_ray(robot_pos, ray_dir, obs_x, obs_y)

            # Consider the robot's radius for clearance
            safe_intersect_t = intersect_t - self.robot_radius * self.safety_distance_ratio
            safe_intersect_t = max(0, safe_intersect_t) # Ensure non-negative

            # Also check against environment boundaries along this ray (optional but good)
            boundary_t = float('inf')
             # Left (x=0)
            if abs(ray_dir[0]) > 1e-6 and ray_dir[0] < 0:
                t_bound = -robot_x / ray_dir[0]; boundary_t = min(boundary_t, t_bound)
            # Right (x=width)
            if abs(ray_dir[0]) > 1e-6 and ray_dir[0] > 0:
                t_bound = (self.width - robot_x) / ray_dir[0]; boundary_t = min(boundary_t, t_bound)
            # Bottom (y=0)
            if abs(ray_dir[1]) > 1e-6 and ray_dir[1] < 0:
                t_bound = -robot_y / ray_dir[1]; boundary_t = min(boundary_t, t_bound)
            # Top (y=height)
            if abs(ray_dir[1]) > 1e-6 and ray_dir[1] > 0:
                t_bound = (self.height - robot_y) / ray_dir[1]; boundary_t = min(boundary_t, t_bound)

            final_t = min(safe_intersect_t, boundary_t, self.avoidance_ray_max_dist)
            ray_results.append((norm_angle, final_t))

        # --- Find Gaps and Select Best Direction ---
        gaps = [] # List of tuples (start_angle, end_angle, mid_angle, min_dist_in_gap, goal_angle_diff)
        in_gap = False
        current_gap_start_angle = None
        current_gap_min_dist = float('inf')

        # Add sentinel points to handle gaps at the edges of the scan
        # Wrap around angle slightly beyond the span to close gaps
        start_angle_wrap = self._normalize_angle(ray_results[0][0] - (ray_angles[1]-ray_angles[0]))
        end_angle_wrap = self._normalize_angle(ray_results[-1][0] + (ray_angles[1]-ray_angles[0]))
        # Use a distance indicating blockage for sentinels
        extended_results = [(start_angle_wrap, 0.0)] + ray_results + [(end_angle_wrap, 0.0)]


        for i in range(len(extended_results)):
            angle, dist = extended_results[i]
            is_clear = dist >= self.avoidance_ray_max_dist * 0.95 # Consider a ray clear if it reaches near max distance

            if is_clear and not in_gap:
                # Start of a new gap
                in_gap = True
                current_gap_start_angle = angle
                current_gap_min_dist = dist
            elif is_clear and in_gap:
                # Continue in gap, update min distance found within it
                current_gap_min_dist = min(current_gap_min_dist, dist)
            elif not is_clear and in_gap:
                # End of the gap
                in_gap = False
                gap_end_angle = extended_results[i-1][0] # Last clear angle
                # Calculate gap properties
                # Ensure start/end are ordered correctly after normalization potentially crossing -pi/pi
                norm_start = self._normalize_angle(current_gap_start_angle)
                norm_end = self._normalize_angle(gap_end_angle)

                # Use angular difference calculation that handles wrap-around for midpoint
                diff = self._normalize_angle(norm_end - norm_start)
                mid_angle = self._normalize_angle(norm_start + diff / 2.0)

                # Calculate difference between gap midpoint and the goal direction
                goal_diff = abs(self._normalize_angle(mid_angle - angle_to_goal))

                gaps.append((norm_start, norm_end, mid_angle, current_gap_min_dist, goal_diff))
                current_gap_start_angle = None
                current_gap_min_dist = float('inf')

        if not gaps:
             # No clear gaps found within the scanned range and distance!
             # This is problematic. Maybe the obstacle is very large or robot is cornered.
             # FIXED: Make sure we're moving AWAY from the obstacle, not towards it
             print("Avoidance: No clear gap found by ray casting! Falling back to direct avoidance.")
             
             # Get vector from obstacle to robot
             vec_to_robot = np.array([robot_x - obs_x, robot_y - obs_y])
             
             # If we're at the obstacle center, move in the opposite direction of current orientation
             if np.linalg.norm(vec_to_robot) < 1e-6:
                 print("Avoidance: At obstacle center! Moving in opposite direction.")
                 return self._normalize_angle(robot_orientation + np.pi)
                 
             # Otherwise move directly away from the obstacle
             angle_away = np.arctan2(vec_to_robot[1], vec_to_robot[0])
             print(f"Avoidance: Moving away from obstacle at angle {angle_away:.2f}")
             return self._normalize_angle(angle_away)


        # Select the best gap (closest midpoint angle to the goal angle)
        gaps.sort(key=lambda x: x[4]) # Sort by goal_angle_diff
        best_gap = gaps[0]
        start_angle, end_angle, mid_angle, _, _ = best_gap 

        chosen_avoidance_angle = mid_angle # Start with the midpoint angle
        try:
            # Find the distance of boundary rays from ray_results
            # Note: Need to handle case where angles might not match exactly due to linspace
            start_entry = min(ray_results, key=lambda x: abs(self._normalize_angle(x[0] - start_angle)))
            end_entry = min(ray_results, key=lambda x: abs(self._normalize_angle(x[0] - end_angle)))
            start_dist = start_entry[1]
            end_dist = end_entry[1]

            critical_distance = self.robot_radius * 3.0 # Distance threshold (Needs tuning)
            max_bias_angle = np.deg2rad(20.0) # Maximum bias angle (Needs tuning)
            bias_angle = 0.0

            # Calculate bias if boundary rays are too close
            if start_dist < critical_distance:
                bias_factor = (1.0 - np.clip(start_dist / critical_distance, 0, 1))**2 # Square for stronger bias when very close
                bias_angle += bias_factor * max_bias_angle # Bias away from start_angle (increase angle)

            if end_dist < critical_distance:
                bias_factor = (1.0 - np.clip(end_dist / critical_distance, 0, 1))**2 # Square for stronger bias when very close
                bias_angle -= bias_factor * max_bias_angle # Bias away from end_angle (decrease angle)

            chosen_avoidance_angle = self._normalize_angle(mid_angle + bias_angle)

            # (Optional) Ensure final angle is still reasonably within the original gap
            # For example: check if it deviates too much from the middle of the gap
            angle_diff_gap = abs(self._normalize_angle(end_angle - start_angle))
            angle_diff_final = abs(self._normalize_angle(chosen_avoidance_angle - mid_angle))
            if angle_diff_final > angle_diff_gap * 0.6: # If deviation is too large, may limit it
                 # Limit bias to not go too far from midpoint, e.g., limit to 50% distance to boundary
                 # Or simply accept the result if the bias calculation is reasonable
                 pass # Currently accept the result of bias


        except Exception as e:
            print(f"Error during avoidance angle biasing: {e}")
            # If error occurs, fall back to using the original midpoint angle
            chosen_avoidance_angle = mid_angle

        # Optional: Debug print
        print(f"Avoidance: Found Gap ({best_gap[0]:.2f} to {best_gap[1]:.2f}), Mid={best_gap[2]:.2f}, Goal={angle_to_goal:.2f}, Chosen={chosen_avoidance_angle:.2f}")

        return self._normalize_angle(chosen_avoidance_angle)


# --- END OF FILE ray_tracing_algorithm.py ---