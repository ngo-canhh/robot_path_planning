# --- START OF FILE bubble_rebound_v2.py ---

import numpy as np
import math
from components.obstacle import Obstacle
from components.shape import Shape

class BubbleRebound:
    def __init__(self, env_width: float, env_height: float, env_dt: float, num_rays: int, robot_radius: float, sensor_range: float, K_values: list[float] = None):
        """
        Initializes the Bubble Rebound calculator.

        Args:
            env_width: Width of the environment.
            env_height: Height of the environment.
            env_dt: Time step of the environment/controller update.
            num_rays: Number of virtual sensor rays to cast.
            robot_radius: Radius of the robot.
            sensor_range: Maximum range of the virtual sensors.
            K_values: List of Ki constants for each ray, controlling bubble boundary.
                      If None, defaults to 1.0 for all rays.
        """
        if num_rays <= 0:
            raise ValueError("Number of rays must be positive.")
        if env_dt <= 0:
            # Warn or raise error if dt is invalid, using a default might hide issues
            print("Warning: env_dt is non-positive, using default 0.1s for bubble calculation.")
            self.env_dt = 0.1
        else:
            self.env_dt = env_dt

        self.num_rays = num_rays
        self.robot_radius = robot_radius
        self.sensor_range = sensor_range
        self.width = env_width
        self.height = env_height
        self.angular_step = math.pi / self.num_rays # Assuming 180 degree span centered forward
        self.epsilon = 1e-9 # Small value to prevent division by zero

        # Initialize K values (bubble boundary constants per ray)
        if K_values is not None:
            if len(K_values) != num_rays:
                raise ValueError(f"Length of K_values must be {num_rays}")
            self.K = np.array(K_values)
        else:
            self.K = np.ones(num_rays)

        # Precompute ray angles relative to robot's forward direction
        # Angles range from -pi/2 to +pi/2 relative to forward
        self.relative_ray_angles = np.linspace(-math.pi / 2, math.pi / 2, self.num_rays)


    def set_K(self, K_values: list[float]):
        """Sets the Ki constants for the bubble boundary calculation."""
        if len(K_values) != self.num_rays:
            raise ValueError(f"Length of K_values must be {self.num_rays}")
        self.K = np.array(K_values)

    def _get_ray_intersection_distance(self, robot_pos: np.ndarray, ray_dir: np.ndarray, obstacles: list[Obstacle]) -> float:
        """Calculates the closest intersection distance for a single ray."""
        closest_t = self.sensor_range

        # 1. Check intersections with obstacles
        for obs in obstacles:
            obs_pos = np.array(obs.get_position()) # Use the obstacle's reference position
            # Note: intersect_ray in Shape should handle obs_pos internally if needed,
            # or accept it as an argument. Assuming it takes obs_x, obs_y here.
            t = obs.shape.intersect_ray(robot_pos, ray_dir, obs_pos[0], obs_pos[1])

            # We need the distance to the *actual* hit point, not an "effective" distance yet
            # We also need to consider the robot radius later, perhaps by checking if t < bubble_boundary
            if 0 < t < closest_t: # Check 0 < t to avoid intersections behind the robot
                 closest_t = t

        # 2. Check intersections with environment boundaries
        boundary_t = self.sensor_range
        # Check only if ray component points towards the boundary
        # Left (x=0)
        if ray_dir[0] < -self.epsilon:
            t_bound = -robot_pos[0] / ray_dir[0]
            if 0 <= t_bound < boundary_t: boundary_t = t_bound
        # Right (x=width)
        if ray_dir[0] > self.epsilon:
            t_bound = (self.width - robot_pos[0]) / ray_dir[0]
            if 0 <= t_bound < boundary_t: boundary_t = t_bound
        # Bottom (y=0)
        if ray_dir[1] < -self.epsilon:
            t_bound = -robot_pos[1] / ray_dir[1]
            if 0 <= t_bound < boundary_t: boundary_t = t_bound
        # Top (y=height)
        if ray_dir[1] > self.epsilon:
            t_bound = (self.height - robot_pos[1]) / ray_dir[1]
            if 0 <= t_bound < boundary_t: boundary_t = t_bound

        # Final distance is the minimum of obstacle hit, boundary hit, or sensor range
        final_t = min(closest_t, boundary_t, self.sensor_range)

        # Ensure distance is non-negative
        return max(0.0, final_t)


    def compute_rebound_angle(self, robot_x: float, robot_y: float, robot_orientation: float, robot_velocity: float, obstacles: list[Obstacle]):
        """
        Computes the rebound angle based on the Bubble Rebound algorithm.

        Args:
            robot_x, robot_y: Current robot position.
            robot_orientation: Current robot orientation angle (in radians).
            robot_velocity: Current robot scalar velocity.
            obstacles: List of perceived obstacles (StaticObstacle or DynamicObstacle).

        Returns:
            tuple: (rebound_angle, is_rebound_active)
                   rebound_angle (float): The calculated rebound angle (in radians).
                                           Defaults to robot_orientation if no rebound needed.
                   is_rebound_active (bool): True if any obstacle was detected within its bubble boundary.
        """
        robot_pos = np.array([robot_x, robot_y])
        sum_weighted_angles = 0.0
        sum_weights = 0.0
        is_rebound_active = False
        min_distance = float('inf')

        # Calculate bubble boundaries for all rays first
        # Paper definition: Kᵢ * V * Δt. This represents a distance threshold.
        bubble_boundaries = self.K * abs(robot_velocity) * self.env_dt # abs(V) ensures non-negative boundary

        # Iterate through each virtual ray
        for i in range(self.num_rays):
            # Calculate absolute angle of the ray in world frame
            absolute_ray_angle = robot_orientation + self.relative_ray_angles[i]
            # Ensure angle is normalized (optional but good practice)
            # absolute_ray_angle = math.atan2(math.sin(absolute_ray_angle), math.cos(absolute_ray_angle))

            ray_dir = np.array([math.cos(absolute_ray_angle), math.sin(absolute_ray_angle)])

            # Get the measured distance along this ray (Di in the paper)
            measured_distance = self._get_ray_intersection_distance(robot_pos, ray_dir, obstacles) - self.robot_radius

            # --- Check if this ray hits something within its bubble boundary ---
            # Paper: "If an obstacle is detected within the sensitivity bubble..."
            # The bubble check determines IF rebound is needed.
            # The measured_distance 'Dᵢ' determines the WEIGHT in the rebound angle calculation.
            # We need to compare measured_distance to the bubble boundary, possibly considering robot radius.

            # Option 1: Compare direct measurement to bubble boundary (as in paper code snippet)
            # bubble_check_distance = bubble_boundaries[i]
            # Option 2: Compare distance-to-boundary to bubble boundary (more physically intuitive?)
            # distance_to_boundary = max(0.0, measured_distance - self.robot_radius)
            # bubble_check_distance = bubble_boundaries[i]

            # Let's stick closer to the paper's interpretation for now: compare raw distance reading.
            if measured_distance < bubble_boundaries[i]:
                is_rebound_active = True
                # print(f"Ray {i}: Hit within bubble! Dist={measured_distance:.2f}, Boundary={bubble_boundaries[i]:.2f}") # Debug

            # --- Accumulate for rebound angle calculation (using ALL rays) ---
            # Weight = measured_distance (Di). Angle = absolute_ray_angle (αi).
            # Paper formula: AR = Σ(αi * Di) / Σ(Di)
            weight = measured_distance
            sum_weighted_angles += absolute_ray_angle * weight
            sum_weights += weight

            min_distance = min(min_distance, measured_distance)

        # Calculate the final rebound angle (AR)
        if is_rebound_active and sum_weights > self.epsilon:
            rebound_angle = sum_weighted_angles / sum_weights
            # Normalize the resulting angle
            rebound_angle = math.atan2(math.sin(rebound_angle), math.cos(rebound_angle))
            # print(f"Rebound Active! Angle: {math.degrees(rebound_angle):.1f}") # Debug
        else:
            # No rebound needed or weights are zero, maintain current orientation or target orientation
            # Returning current orientation makes sense if this function ONLY calculates rebound
            rebound_angle = robot_orientation # Or potentially target_orientation if available?
            # print("Rebound NOT Active.") # Debug


        return rebound_angle, is_rebound_active, min_distance

# --- END OF FILE bubble_rebound_v2.py ---