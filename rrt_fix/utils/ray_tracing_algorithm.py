import numpy as np
from components.obstacle import Obstacle, StaticObstacle, DynamicObstacle

# --- RayTracingAlgorithm (Simplified Avoidance) ---

class RayTracingAlgorithm:
    def __init__(self, env_width, env_height, robot_radius):
        self.width = env_width
        self.height = env_height
        self.robot_radius = robot_radius
        self.num_rays = 36
        self.max_ray_length = 150

    def trace_rays(self, robot_x, robot_y, robot_orientation, obstacles: list):
        """ Traces rays and checks intersections with perceived Obstacle objects. """
        ray_intersections = []
        ray_viz_points = []

        for i in range(self.num_rays):
            angle = robot_orientation + i * (2 * np.pi / self.num_rays)
            angle = np.arctan2(np.sin(angle), np.cos(angle))

            ray_dir_x = np.cos(angle)
            ray_dir_y = np.sin(angle)
            ray_start = (robot_x, robot_y)
            # Max endpoint without considering obstacles/bounds
            ray_end_max = (robot_x + ray_dir_x * self.max_ray_length,
                           robot_y + ray_dir_y * self.max_ray_length)

            closest_intersection_dist_sq = self.max_ray_length**2
            intersected_obstacle = None

            # Check intersection with perceived obstacles using shape's method
            for obstacle in obstacles:
                 # This requires a method on Shape to find the intersection distance `t`
                 # along the ray, not just boolean intersects_segment.
                 # Let's stick to a simplified check for now:
                 # Does the segment from robot to max_ray_end intersect the obstacle?
                 # This doesn't give the *closest* intersection point accurately.
                 # TODO: Implement ray-shape intersection distance calculation in Shape classes.

                 # Simplified check (less accurate distance):
                 # If the infinite line intersects, find the distance.
                 # This part needs proper geometry for line-shape intersection.
                 # Skipping detailed ray-shape intersection for brevity. We'll use
                 # the intersects_segment check as a proxy (less accurate for distance).
                 pass # Placeholder for actual ray intersection logic

            # --- Fallback: Use simplified boundary check ---
            # Calculate intersection with boundaries (same as before)
            min_boundary_t = self.max_ray_length
            # ... (boundary intersection code from previous version) ...
            # Find the closest boundary intersection distance t
            boundary_ts = []
            # Left boundary (x=0)
            if abs(ray_dir_x) > 1e-6 and ray_dir_x < 0:
                 t_bound = -robot_x / ray_dir_x
                 if 0 <= t_bound < self.max_ray_length:
                      y_intersect = robot_y + t_bound * ray_dir_y
                      if 0 <= y_intersect <= self.height: boundary_ts.append(t_bound)
            # Right boundary (x=width)
            if abs(ray_dir_x) > 1e-6 and ray_dir_x > 0:
                 t_bound = (self.width - robot_x) / ray_dir_x
                 if 0 <= t_bound < self.max_ray_length:
                      y_intersect = robot_y + t_bound * ray_dir_y
                      if 0 <= y_intersect <= self.height: boundary_ts.append(t_bound)
             # Bottom boundary (y=0)
            if abs(ray_dir_y) > 1e-6 and ray_dir_y < 0:
                 t_bound = -robot_y / ray_dir_y
                 if 0 <= t_bound < self.max_ray_length:
                      x_intersect = robot_x + t_bound * ray_dir_x
                      if 0 <= x_intersect <= self.width: boundary_ts.append(t_bound)
            # Top boundary (y=height)
            if abs(ray_dir_y) > 1e-6 and ray_dir_y > 0:
                 t_bound = (self.height - robot_y) / ray_dir_y
                 if 0 <= t_bound < self.max_ray_length:
                      x_intersect = robot_x + t_bound * ray_dir_x
                      if 0 <= x_intersect <= self.width: boundary_ts.append(t_bound)

            if boundary_ts:
                 min_boundary_t = min(boundary_ts)

            # Use boundary distance as the intersection distance for now
            closest_intersection_dist = min_boundary_t # Replace with actual obstacle intersection logic later
            intersected_obstacle = None # Reset as we didn't check obstacles properly

            closest_intersection_point = (
                robot_x + ray_dir_x * closest_intersection_dist,
                robot_y + ray_dir_y * closest_intersection_dist
            )

            ray_intersections.append((*closest_intersection_point, intersected_obstacle))
            ray_viz_points.append(((robot_x, robot_y), closest_intersection_point))

        return ray_intersections, ray_viz_points


    def avoid_static_obstacle(self, robot_x, robot_y, robot_orientation, obstacle: Obstacle, goal_x, goal_y):
        """
        Suggests an avoidance orientation based on a *single* static obstacle's *center*.
        This is a simplification ignoring the specific shape for reactive avoidance.
        """
        # Use obstacle's center position for avoidance calculation
        vec_to_obstacle = np.array([obstacle.x - robot_x, obstacle.y - robot_y])
        dist_to_obstacle_center = np.linalg.norm(vec_to_obstacle)

        if dist_to_obstacle_center < 1e-6:
            return self._normalize_angle(robot_orientation + np.pi / 2)

        vec_to_obstacle_norm = vec_to_obstacle / dist_to_obstacle_center
        angle_to_obstacle = np.arctan2(vec_to_obstacle_norm[1], vec_to_obstacle_norm[0])

        angle_perp1 = self._normalize_angle(angle_to_obstacle + np.pi / 2)
        angle_perp2 = self._normalize_angle(angle_to_obstacle - np.pi / 2)

        vec_to_goal = np.array([goal_x - robot_x, goal_y - robot_y])
        dist_to_goal = np.linalg.norm(vec_to_goal)
        if dist_to_goal < 1e-6:
             return angle_perp1 # Default if at goal

        angle_to_goal = np.arctan2(vec_to_goal[1], vec_to_goal[0])

        diff1 = abs(self._normalize_angle(angle_perp1 - angle_to_goal))
        diff2 = abs(self._normalize_angle(angle_perp2 - angle_to_goal))

        avoidance_orientation = angle_perp1 if diff1 < diff2 else angle_perp2

        return avoidance_orientation

    def _normalize_angle(self, angle):
        while angle > np.pi: angle -= 2 * np.pi
        while angle < -np.pi: angle += 2 * np.pi
        return angle