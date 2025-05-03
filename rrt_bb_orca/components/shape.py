# --- START OF REVISED shape.py SECTION ---

import math
import numpy as np
import random
from abc import ABC, abstractmethod
import matplotlib.patches as patches
from typing import List, Tuple
from matplotlib import pyplot as plt
from matplotlib.patches import Circle as mplCircle
from matplotlib.patches import Rectangle as mplRectangle
from matplotlib.patches import Polygon as mplPolygon
from matplotlib.path import Path # Important for polygon checks

# --- Shape Definitions (Keep Shape ABC, Circle, Rectangle as before) ---

class Shape(ABC):
    """Abstract base class for geometric shapes used in obstacles."""

    @abstractmethod
    def check_collision(self, robot_x: float, robot_y: float, robot_radius: float, obs_x: float, obs_y: float) -> bool:
        """
        Checks if a robot (circle) collides with this shape instance positioned at (obs_x, obs_y).

        Args:
            robot_x: Robot's center x-coordinate.
            robot_y: Robot's center y-coordinate.
            robot_radius: Robot's radius.
            obs_x: Obstacle's reference point x-coordinate (e.g., centroid).
            obs_y: Obstacle's reference point y-coordinate (e.g., centroid).

        Returns:
            True if collision occurs, False otherwise.
        """
        pass

    @abstractmethod
    def intersects_segment(self, p1: tuple, p2: tuple, robot_radius: float, obs_x: float, obs_y: float) -> bool:
        """
        Checks if a line segment (p1 to p2), potentially thickened by robot_radius,
        intersects this shape instance positioned at (obs_x, obs_y).
        Used primarily for planning checks where the path segment needs clearance.

        Args:
            p1: Tuple (x, y) representing the start of the segment.
            p2: Tuple (x, y) representing the end of the segment.
            robot_radius: Robot's radius (acts as padding).
            obs_x: Obstacle's reference point x-coordinate.
            obs_y: Obstacle's reference point y-coordinate.

        Returns:
            True if intersection occurs, False otherwise.
        """
        pass

    @abstractmethod
    def get_patch(self, center_x: float, center_y: float, color: str, alpha: float, zorder: int) -> patches.Patch:
        """
        Creates a Matplotlib patch representation of the shape.

        Args:
            center_x: X-coordinate for the shape's center/reference point.
            center_y: Y-coordinate for the shape's center/reference point.
            color: Fill color for the patch.
            alpha: Transparency alpha value.
            zorder: Drawing order.

        Returns:
            A matplotlib.patches.Patch object.
        """
        pass

    @abstractmethod
    def get_observation_params(self) -> tuple:
        """
        Returns parameters defining the shape for the observation space.
        Format: (shape_type_enum, (param1, param2, param3, ...))
        shape_type_enum: 0=Circle, 1=Rectangle, ...
        param1, param2, param3, ...: Shape-specific dimensions (e.g., radius, width, height, angle). Number of params depend on shape type.
        """
        pass

    @abstractmethod
    def get_efficient_distance(self, point_x: float, point_y: float, obs_x: float, obs_y: float) -> float:
        """
        Returns the shortest distance from the point (point_x, point_y) to the boundary of the shape.
        Distance is positive if the point is outside, zero if on the boundary, and negative if inside.
        (Note: Original docstring slightly ambiguous, clarifying here)

        Args:
            point_x: X-coordinate of the point.
            point_y: Y-coordinate of the point.
            obs_x: Obstacle's reference point x-coordinate.
            obs_y: Obstacle's reference point y-coordinate.

        Returns:
            Signed distance from the point to the shape boundary.
        """
        pass

    @abstractmethod
    def get_centroid(self) -> np.ndarray:
        """ Returns the centroid relative to the shape's origin (usually 0,0). """
        pass # Might not be needed if obs_x, obs_y is always the centroid

    @abstractmethod
    def get_effective_radius(self) -> float:
        """
        Returns the effective radius of the shape for collision checks.
        For circles, this is the radius. For rectangles, it can be the diagonal or a similar metric.
        """
        pass

    @abstractmethod
    def get_effective_vector(self, point_x: float, point_y: float, obs_x: float, obs_y: float) -> float:
        """
        Returns the shortest vector from point to the rectangle's boundary.
        
        Args:
            point_x, point_y: Coordinates of the point
            obs_x, obs_y: Coordinates of the rectangle's center
            
        Returns:
            Vector from point to closest point on rectangle boundary (as numpy array)
        """
        pass

    @abstractmethod
    def intersect_ray(self, ray_origin: np.ndarray, ray_direction: np.ndarray, obs_x: float, obs_y: float) -> float:
        """
        Calculates the intersection distance of a ray with this shape instance.

        Args:
            ray_origin: np.ndarray (x, y), starting point of the ray.
            ray_direction: np.ndarray (dx, dy), unit direction vector of the ray.
            obs_x: Obstacle's reference point x-coordinate.
            obs_y: Obstacle's reference point y-coordinate.

        Returns:
            float: The distance 't' along the ray direction where the *first* intersection
                   occurs (P_intersect = ray_origin + t * ray_direction).
                   Returns float('inf') if there is no intersection.
        """
        pass


    @classmethod
    def create_random_shape(cls, seed=None, shape_name=None):
        """
        Creates a random shape based on seed or name.
        
        Args:
            seed: Random seed for deterministic shape generation.
            shape_name: Optional name to determine which shape to create.
                        If provided, takes precedence over random selection.
        
        Returns:
            A random instance of a Shape subclass.
        """
        # Set the random seed if provided
        if seed is not None:
            random.seed(seed)
            
        # If shape_name is provided, use it to determine the shape
        if shape_name is not None:
            shape_name = shape_name.lower()
            if "circle" in shape_name:
                radius = random.uniform(15, 40)
                return Circle(radius)
            elif "rectangle" in shape_name or "rect" in shape_name:
                width = random.uniform(20, 60)
                height = random.uniform(20, 60)
                angle = random.uniform(-np.pi/4, np.pi/4)
                return Rectangle(width, height, angle)
            elif "triangle" in shape_name:
                # Generate a random triangle within a bounded area
                vertices = []
                for _ in range(3):
                    x = random.uniform(-60, 60)
                    y = random.uniform(-60, 60)
                    vertices.append((x, y))
                return Triangle(vertices)
            elif "polygon" in shape_name:
                # Generate a random convex polygon
                n_vertices = random.randint(4, 15)
                vertices = []
                for i in range(n_vertices):
                    angle = 2 * math.pi * i / n_vertices
                    radius = random.uniform(0.5, 2.0)
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    # Add some noise to make it less regular
                    x += random.uniform(-60, 60)
                    y += random.uniform(-60, 60)
                    vertices.append((x, y))
                return Polygon(vertices)
        
        # If no name provided or name not recognized, choose randomly
        shape_type = random.choice(["circle", "rectangle", "triangle", "polygon"])
        return cls.create_random_shape(seed=None, shape_name=shape_type)



class Circle(Shape):
    """Circular Shape."""
    SHAPE_TYPE_ENUM = 0

    def __init__(self, radius: float):
        if radius <= 0:
            raise ValueError("Circle radius must be positive")
        self.radius = radius

    def check_collision(self, robot_x: float, robot_y: float, robot_radius: float, obs_x: float, obs_y: float) -> bool:
        """Checks collision between robot circle and obstacle circle."""
        min_dist_sq = (self.radius + robot_radius)**2
        dist_sq = (robot_x - obs_x)**2 + (robot_y - obs_y)**2
        # Use a small tolerance for floating point comparisons if necessary
        # return dist_sq < min_dist_sq - 1e-9
        return dist_sq < min_dist_sq

    def intersects_segment(self, p1: tuple, p2: tuple, robot_radius: float, obs_x: float, obs_y: float) -> bool:
        """Checks if segment intersects circle (with padding)."""
        effective_radius = self.radius + robot_radius
        effective_radius_sq = effective_radius**2
        p1_arr = np.array(p1)
        p2_arr = np.array(p2)
        obs_pos = np.array([obs_x, obs_y])

        # Check if endpoints are inside the padded circle
        if np.sum((p1_arr - obs_pos)**2) < effective_radius_sq:
            return True
        if np.sum((p2_arr - obs_pos)**2) < effective_radius_sq:
            return True

        segment_vec = p2_arr - p1_arr
        segment_len_sq = np.dot(segment_vec, segment_vec)

        # If segment is effectively a point (and already checked)
        if segment_len_sq < 1e-12:
            return False # Endpoints already checked

        # Project obstacle center onto the line defined by the segment
        # t = dot(obs_pos - p1, p2 - p1) / |p2 - p1|^2
        t = np.dot(obs_pos - p1_arr, segment_vec) / segment_len_sq

        # Find the closest point on the *line segment* to the obstacle center
        if t < 0.0:
            closest_point_on_segment = p1_arr
        elif t > 1.0:
            closest_point_on_segment = p2_arr
        else:
            # Closest point is within the segment projection
            closest_point_on_segment = p1_arr + t * segment_vec

        # Check distance squared from obstacle center to this closest point on the segment
        dist_sq_to_segment = np.sum((obs_pos - closest_point_on_segment)**2)

        return dist_sq_to_segment < effective_radius_sq

    def get_patch(self, center_x: float, center_y: float, color: str, alpha: float, zorder: int) -> patches.Patch:
        # Use mplCircle directly for clarity, though patches.Circle is the same
        return mplCircle((center_x, center_y), self.radius, fc=color, alpha=alpha, zorder=zorder)

    def get_observation_params(self) -> tuple:
        """
        Format: (0, (radius)) 
        """
        return (Circle.SHAPE_TYPE_ENUM, (self.radius))

    def get_efficient_distance(self, point_x, point_y, obs_x, obs_y):
        # Distance from point to circle center minus radius
        dist_to_center = math.sqrt((point_x - obs_x)**2 + (point_y - obs_y)**2)
        return dist_to_center - self.radius

    def get_centroid(self) -> np.ndarray:
        return np.array([0.0, 0.0])

    def get_effective_radius(self) -> float:
        return self.radius

    def get_effective_vector(self, point_x: float, point_y: float, obs_x: float, obs_y: float) -> np.ndarray:
        """
        Returns the shortest vector from point to the circle's boundary.
        
        Args:
            point_x, point_y: Coordinates of the point
            obs_x, obs_y: Coordinates of the circle's center
            
        Returns:
            Vector from point to closest point on circle boundary (as numpy array)
        """
        # Vector from circle center to point
        direction = np.array([point_x - obs_x, point_y - obs_y])
        
        # Distance from circle center to point
        distance = np.linalg.norm(direction)
        
        if distance < 1e-9:  # Point is at circle center
            # Return arbitrary direction with radius length
            return np.array([self.radius, 0.0])
        
        # Normalize the direction vector
        normalized_direction = direction / distance
        
        # Calculate closest point on circle boundary
        closest_point = obs_x + normalized_direction[0] * self.radius, obs_y + normalized_direction[1] * self.radius
        
        # Return vector from point to closest point on boundary
        return np.array([closest_point[0] - point_x, closest_point[1] - point_y])
    
    def intersect_ray(self, ray_origin: np.ndarray, ray_direction: np.ndarray, obs_x: float, obs_y: float) -> float:
        """
        Calculates the intersection distance of a ray with this circle instance.
        Ray equation: P(t) = ray_origin + t * ray_direction
        Circle equation: |P - C|^2 = r^2, where C is (obs_x, obs_y)

        Args:
            ray_origin: np.ndarray (x, y), starting point of the ray.
            ray_direction: np.ndarray (dx, dy), unit direction vector of the ray.
                         (Should be normalized for 't' to represent distance).
            obs_x: Circle's center x-coordinate.
            obs_y: Circle's center y-coordinate.

        Returns:
            float: The distance 't' along the ray direction where the *first* positive
                   intersection occurs (P_intersect = ray_origin + t * ray_direction).
                   Returns float('inf') if there is no intersection in the forward direction (t >= 0).
        """
        circle_center = np.array([obs_x, obs_y])
        # Vector from circle center to ray origin
        oc = ray_origin - circle_center

        # Coefficients of the quadratic equation: a*t^2 + b*t + c = 0
        # a = D · D (should be 1 if D is normalized, but calculate for robustness)
        a = np.dot(ray_direction, ray_direction)
        # b = 2 * (OC · D)
        b = 2.0 * np.dot(oc, ray_direction)
        # c = OC · OC - r^2
        c = np.dot(oc, oc) - self.radius * self.radius

        # Calculate the discriminant: delta = b^2 - 4ac
        discriminant = b*b - 4.0*a*c

        # Check if the ray misses the circle (no real roots)
        if discriminant < 0:
            return float('inf')

        # Calculate the two potential solutions for t
        # Add small epsilon to prevent division by zero if a is extremely small,
        # although this shouldn't happen if ray_direction is properly normalized.
        denom = 2.0 * a
        if abs(denom) < 1e-9:
             # This implies ray_direction is near zero vector, which is invalid.
             # Or the ray origin is exactly on the circle and direction is tangent?
             # Handle as no intersection for simplicity, as input is likely invalid.
              return float('inf')

        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / denom
        t2 = (-b + sqrt_discriminant) / denom

        # We want the smallest non-negative intersection time 't'.
        # If t1 >= 0, it's the first intersection point (or tangent point).
        if t1 >= -1e-9: # Use small tolerance for t >= 0 check
            return t1
        # If t1 is negative, but t2 is non-negative, the ray started inside the circle.
        # The first intersection *in the direction of the ray* is the exit point t2.
        elif t2 >= -1e-9:
             # Ray started inside, t2 is the exit distance
             return t2
        # If both t1 and t2 are negative, the circle is entirely behind the ray origin.
        else:
            return float('inf')


class Rectangle(Shape):
    """Rectangular Shape, potentially rotated."""
    SHAPE_TYPE_ENUM = 1

    def __init__(self, width: float, height: float, angle: float = 0.0):
        """
        Args:
            width: Width along the rectangle's local x-axis.
            height: Height along the rectangle's local y-axis.
            angle: Rotation angle in radians relative to the world x-axis.
                   The rectangle is centered at (obs_x, obs_y).
        """
        if width <= 0 or height <= 0:
            raise ValueError("Rectangle width and height must be positive")
        self.width = width
        self.height = height
        self.angle = angle # Angle in radians
        self._cos_a = math.cos(angle)
        self._sin_a = math.sin(angle)
        self._half_w = width / 2.0
        self._half_h = height / 2.0
        self.local_bounds_min = np.array([-self._half_w, -self._half_h])
        self.local_bounds_max = np.array([self._half_w, self._half_h])

    def _world_to_local(self, point_x: float, point_y: float, obs_x: float, obs_y: float) -> Tuple[float, float]:
        """Transforms world coordinates to the rectangle's local frame (centered at origin)."""
        dx = point_x - obs_x
        dy = point_y - obs_y
        local_x = dx * self._cos_a + dy * self._sin_a
        local_y = -dx * self._sin_a + dy * self._cos_a
        return local_x, local_y

    def _local_to_world(self, local_x: float, local_y: float, obs_x: float, obs_y: float) -> Tuple[float, float]:
        """Transforms rectangle's local coordinates back to world frame."""
        world_dx = local_x * self._cos_a - local_y * self._sin_a
        world_dy = local_x * self._sin_a + local_y * self._cos_a
        return obs_x + world_dx, obs_y + world_dy

    def _get_world_corners(self, obs_x: float, obs_y: float) -> List[Tuple[float, float]]:
        """ Get world coordinates of the rectangle corners. """
        corners_local = [
            (self._half_w, self._half_h),
            (-self._half_w, self._half_h),
            (-self._half_w, -self._half_h),
            (self._half_w, -self._half_h)
        ]
        corners_world = [self._local_to_world(lx, ly, obs_x, obs_y) for lx, ly in corners_local]
        return corners_world

    def check_collision(self, robot_x: float, robot_y: float, robot_radius: float, obs_x: float, obs_y: float) -> bool:
        """Checks collision using distance from robot center to rectangle boundary."""
        # Transform robot center to rectangle's local coordinates
        local_rx, local_ry = self._world_to_local(robot_x, robot_y, obs_x, obs_y)

        # Find the closest point on the solid rectangle (in local coordinates) to the local robot center
        # This point is found by clamping the local coordinates to the rectangle bounds.
        clamped_x = max(-self._half_w, min(self._half_w, local_rx))
        clamped_y = max(-self._half_h, min(self._half_h, local_ry))

        # Calculate the distance squared between the local robot center and this closest point
        dist_sq = (local_rx - clamped_x)**2 + (local_ry - clamped_y)**2

        # Collision occurs if this distance is less than the robot radius
        # Using squared distance avoids sqrt
        return dist_sq < robot_radius**2

    def intersects_segment(self, p1: tuple, p2: tuple, robot_radius: float, obs_x: float, obs_y: float) -> bool:
        """Checks if segment (thickened by robot_radius) intersects rectangle. Uses sampling."""
        # Approximate by sampling points along the segment and checking collision
        # for each point (treated as a robot center) with the rectangle.
        from_arr = np.array(p1)
        to_arr = np.array(p2)
        vec = to_arr - from_arr
        dist = np.linalg.norm(vec)

        if dist < 1e-9: # Segment is essentially a point
             return self.check_collision(p1[0], p1[1], robot_radius, obs_x, obs_y)

        # Determine number of checks based on robot radius and segment length
        # Ensure at least start, end, and one midpoint are checked
        # Sample roughly every half-radius distance along the segment
        num_steps = max(int(dist / (robot_radius * 0.5)) + 1, 2)

        for i in range(num_steps + 1): # Check points including start and end
            t = i / num_steps
            check_point = from_arr + vec * t # Linear interpolation
            # Check collision of this point (as a robot center) with the rectangle
            if self.check_collision(check_point[0], check_point[1], robot_radius, obs_x, obs_y):
                 return True # Collision detected for this sampled point

        return False # No collision found for any checked point

    def get_patch(self, center_x: float, center_y: float, color: str, alpha: float, zorder: int) -> patches.Patch:
        # patches.Rectangle requires bottom-left corner and angle in degrees.
        # Calculate bottom-left corner relative to the center in local coords
        bl_local_x = -self._half_w
        bl_local_y = -self._half_h
        # Transform local bottom-left to world coords
        bl_world_x, bl_world_y = self._local_to_world(bl_local_x, bl_local_y, center_x, center_y)

        # Use mplRectangle directly for clarity
        return mplRectangle((bl_world_x, bl_world_y), self.width, self.height,
                             angle=math.degrees(self.angle), # Patches uses degrees
                             fc=color, alpha=alpha, zorder=zorder)

    def get_observation_params(self) -> tuple:
        """
        Format: (type, (width, height, angle))
        """
        return (Rectangle.SHAPE_TYPE_ENUM, (self.width, self.height, self.angle))

    def get_efficient_distance(self, point_x: float, point_y: float, obs_x: float, obs_y: float) -> float:
        """Calculates signed distance from point to rectangle boundary."""
        # Transform point to local coordinates
        local_x, local_y = self._world_to_local(point_x, point_y, obs_x, obs_y)

        # Distances to the lines forming the rectangle sides
        dx = abs(local_x) - self._half_w
        dy = abs(local_y) - self._half_h

        # Distance calculation based on Voronoi regions of the rectangle
        # See: https://iquilezles.org/articles/distfunctions2d/ (Box distance)

        # External distance (point is outside)
        # max(dx, 0) handles the case where point is inside the x-slab
        # max(dy, 0) handles the case where point is inside the y-slab
        # Distance to corner = sqrt(dx^2 + dy^2) if dx > 0 and dy > 0
        # Distance to edge = dx if dx > 0 and dy <= 0
        # Distance to edge = dy if dy > 0 and dx <= 0
        # Combine using length of vector (max(dx, 0), max(dy, 0))
        external_dist = math.sqrt(max(dx, 0)**2 + max(dy, 0)**2)

        # Internal distance (point is inside)
        # Distance to closest edge = max(dx, dy) where dx, dy are <= 0
        # max(dx, dy) gives the largest (least negative) distance, which is closest edge
        internal_dist = max(dx, dy) # Note: dx, dy are <= 0 if inside

        # Return external distance if point is outside/on boundary (dx>0 or dy>0 or both=0)
        # Return internal distance (which is <= 0) if point is strictly inside (dx<0 and dy<0)
        if dx < 0 and dy < 0:
             # Strictly inside, internal_dist is the negative distance to the closest edge
             return internal_dist
        else:
             # Outside or on the boundary, external_dist is the positive distance
             return external_dist


    def get_centroid(self) -> np.ndarray:
        # Centroid is at the reference point (obs_x, obs_y) by definition
        return np.array([0.0, 0.0])

    def get_effective_radius(self):
        # Radius of the circumscribing circle (distance from center to corner)
        return 0.5 * math.sqrt(self.width**2 + self.height**2)
    
    def get_effective_vector(self, point_x: float, point_y: float, obs_x: float, obs_y: float) -> np.ndarray:
        """
        Returns the shortest vector from point to the rectangle's boundary.
        
        Args:
            point_x, point_y: Coordinates of the point
            obs_x, obs_y: Coordinates of the rectangle's center
            
        Returns:
            Vector from point to closest point on rectangle boundary (as numpy array)
        """
        # Transform point to local coordinates
        local_x, local_y = self._world_to_local(point_x, point_y, obs_x, obs_y)
        
        # Find the closest point on the rectangle boundary
        if -self._half_w <= local_x <= self._half_w and -self._half_h <= local_y <= self._half_h:
            # Point is inside rectangle - find closest edge
            dist_to_right = self._half_w - local_x
            dist_to_left = local_x + self._half_w
            dist_to_top = self._half_h - local_y
            dist_to_bottom = local_y + self._half_h
            
            # Find minimum distance to edge
            min_dist = min(dist_to_right, dist_to_left, dist_to_top, dist_to_bottom)
            
            # Create vector in appropriate direction
            if min_dist == dist_to_right:
                local_closest_x = self._half_w
                local_closest_y = local_y
            elif min_dist == dist_to_left:
                local_closest_x = -self._half_w
                local_closest_y = local_y
            elif min_dist == dist_to_top:
                local_closest_x = local_x
                local_closest_y = self._half_h
            else:  # min_dist == dist_to_bottom
                local_closest_x = local_x
                local_closest_y = -self._half_h
        else:
            # Point is outside rectangle - clamp to edges
            local_closest_x = max(-self._half_w, min(self._half_w, local_x))
            local_closest_y = max(-self._half_h, min(self._half_h, local_y))
        
        # Transform closest point back to world coordinates
        world_closest_x, world_closest_y = self._local_to_world(local_closest_x, local_closest_y, obs_x, obs_y)
        
        # Return vector from point to closest point
        return np.array([world_closest_x - point_x, world_closest_y - point_y])
    
    def _rotate_vector_world_to_local(self, vec_x: float, vec_y: float) -> Tuple[float, float]:
        """Rotates a world vector to the rectangle's local frame."""
        local_x = vec_x * self._cos_a + vec_y * self._sin_a
        local_y = -vec_x * self._sin_a + vec_y * self._cos_a
        return local_x, local_y
    
    def intersect_ray(self, ray_origin: np.ndarray, ray_direction: np.ndarray, obs_x: float, obs_y: float) -> float:
        """
        Calculates the intersection distance of a ray with this rectangle.
        Uses the separating axis theorem (SAT) based approach by transforming
        the ray into the rectangle's local coordinate system (slab method).

        Args:
            ray_origin: np.ndarray (x, y), starting point of the ray in world coords.
            ray_direction: np.ndarray (dx, dy), unit direction vector of the ray in world coords.
            obs_x: Rectangle's center x-coordinate in world coords.
            obs_y: Rectangle's center y-coordinate in world coords.

        Returns:
            float: The distance 't' along the ray direction where the *first* intersection occurs.
                   Returns float('inf') if there is no intersection or intersection is behind the ray origin (t<0).
                   Returns 0.0 if the ray origin is exactly on the boundary.
        """
        # Epsilon for floating point comparisons (division by zero check)
        EPSILON = 1e-9

        # 1. Transform ray origin to rectangle's local coordinates
        local_origin_x, local_origin_y = self._world_to_local(ray_origin[0], ray_origin[1], obs_x, obs_y)
        local_origin = np.array([local_origin_x, local_origin_y])

        # 2. Rotate ray direction vector to rectangle's local coordinates
        local_dir_x, local_dir_y = self._rotate_vector_world_to_local(ray_direction[0], ray_direction[1])
        local_direction = np.array([local_dir_x, local_dir_y])

        # 3. Perform Ray-AABB intersection in local coordinates (Slab method)
        # Rectangle bounds in local coordinates are [-half_w, half_w] and [-half_h, half_h]
        # local_bounds_min = np.array([-self._half_w, -self._half_h]) # Defined in __init__
        # local_bounds_max = np.array([self._half_w, self._half_h]) # Defined in __init__

        t_min = 0.0 # Max of near plane intersections
        t_max = float('inf') # Min of far plane intersections

        for i in range(2): # Iterate through x (i=0) and y (i=1) axes
            # Check for division by zero (ray parallel to slab planes)
            if abs(local_direction[i]) < EPSILON:
                # Ray is parallel to the slab planes for this axis.
                # Check if the origin is outside the slab.
                if local_origin[i] < self.local_bounds_min[i] or local_origin[i] > self.local_bounds_max[i]:
                    return float('inf') # Parallel and outside -> No intersection
                # Otherwise, ray is parallel and inside the slab, continue to next axis
            else:
                # Calculate intersection distances with the slab planes
                t1 = (self.local_bounds_min[i] - local_origin[i]) / local_direction[i]
                t2 = (self.local_bounds_max[i] - local_origin[i]) / local_direction[i]

                # Ensure t1 is intersection with near plane, t2 with far plane
                if t1 > t2:
                    t1, t2 = t2, t1 # Swap

                # Update overall t_min and t_max
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)

                # Check for non-overlapping intervals
                if t_min > t_max:
                    return float('inf') # Box is missed

        # 4. Check if intersection is valid (in front of ray origin)
        # The intersection interval is [t_min, t_max]
        # We are interested in the first intersection point, which is t_min.
        # If t_min > t_max, intervals didn't overlap, already handled.
        # If t_max < 0, the intersection is entirely behind the ray origin.
        if t_max < 0:
             return float('inf')

        # If t_min < 0, the origin is inside the rectangle.
        # The first intersection *in the positive direction* would be t_max.
        # However, the typical definition asks for the first point hit along the ray.
        # If starting inside, the closest intersection point *could* be considered t=0.
        # Let's return t_min if it's non-negative, otherwise infinity (or 0 if starting inside/on boundary is desired)
        # Standard ray tracing often returns the *first positive* t value.
        # If t_min >= 0, it's the first hit distance.
        if t_min >= 0:
            return t_min
        else:
            # Origin is inside. The ray exits at t_max.
            # If we want the distance to the *exit* point when starting inside:
            # return t_max
            # If we want the distance to the *first boundary crossing* (which is behind if t_min < 0):
            # return t_min (but negative is usually ignored)
            # If we want to report "hit" but distance is 0 because we start inside:
            # return 0.0
            # Let's return infinity if the first hit (t_min) is negative,
            # consistent with finding hits strictly in front of the origin.
            return float('inf') # Or return t_max if exit distance is needed when starting inside.


class Triangle(Shape):
    """Triangle Shape."""
    SHAPE_TYPE_ENUM = 2

    def __init__(self, vertices: List[Tuple[float, float]]):
        """
        Initialize a triangle from three vertices.
        
        Args:
            vertices: List of 3 (x, y) coordinates defining the triangle vertices in 
                      the local coordinate system (centered at origin)
        """
        if len(vertices) != 3:
            raise ValueError("Triangle must have exactly 3 vertices")
        
        self.vertices = np.array(vertices)
        
        # Check if vertices are arranged clockwise
        # Compute signed area using shoelace formula
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        area = 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
        
        # If area is positive, vertices are counterclockwise - reverse them
        if area > 0:
            self.vertices = np.flip(self.vertices, axis=0)

    def _point_in_triangle(self, px: float, py: float) -> bool:
        """
        Check if a point (px, py) is inside the triangle using barycentric coordinates.
        
        Args:
            px: x-coordinate of the point
            py: y-coordinate of the point
            
        Returns:
            True if point is inside or on the boundary of triangle, False otherwise
        """
        # Extract vertices
        x1, y1 = self.vertices[0]
        x2, y2 = self.vertices[1]
        x3, y3 = self.vertices[2]
        
        # Compute area of the full triangle
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        
        # Compute areas of three triangles formed by the point and each side
        area1 = 0.5 * abs((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py))
        area2 = 0.5 * abs((x2 - px) * (y3 - py) - (x3 - px) * (y2 - py))
        area3 = 0.5 * abs((x3 - px) * (y1 - py) - (x1 - px) * (y3 - py))
        
        # Point is inside if sum of three areas approximately equals the original area
        return abs(area - (area1 + area2 + area3)) < 1e-9

    def _transform_point(self, px: float, py: float, obs_x: float, obs_y: float) -> Tuple[float, float]:
        """Transform a point from world coordinates to local coordinates."""
        return px - obs_x, py - obs_y

    def _closest_point_on_segment(self, px: float, py: float, ax: float, ay: float, 
                                 bx: float, by: float) -> Tuple[float, float]:
        """
        Find the closest point on a line segment (a-b) to point p.
        
        Args:
            px, py: Coordinates of point p
            ax, ay: Coordinates of segment endpoint a
            bx, by: Coordinates of segment endpoint b
        
        Returns:
            Coordinates of the closest point on the segment
        """
        # Vector from a to b
        abx = bx - ax
        aby = by - ay
        
        # Length squared of segment ab
        ab_len_sq = abx * abx + aby * aby
        
        # If segment is effectively a point, return a
        if ab_len_sq < 1e-12:
            return ax, ay
        
        # Calculate projection of ap onto ab, parameterized as t
        apx = px - ax
        apy = py - ay
        t = max(0, min(1, (apx * abx + apy * aby) / ab_len_sq))
        
        # Return the point on the segment
        return ax + t * abx, ay + t * aby

    def _distance_point_to_segment(self, px: float, py: float, ax: float, ay: float,
                                  bx: float, by: float) -> float:
        """
        Calculate the minimum distance from point p to line segment (a-b).
        
        Returns:
            Distance from point to segment
        """
        # Find closest point on segment
        cx, cy = self._closest_point_on_segment(px, py, ax, ay, bx, by)
        
        # Return distance to this point
        return math.sqrt((px - cx)**2 + (py - cy)**2)

    def check_collision(self, robot_x: float, robot_y: float, robot_radius: float, 
                        obs_x: float, obs_y: float) -> bool:
        """Checks collision between robot circle and triangle."""
        # Transform robot center to local coordinates
        local_rx, local_ry = self._transform_point(robot_x, robot_y, obs_x, obs_y)
        
        # Check if robot center is inside triangle
        if self._point_in_triangle(local_rx, local_ry):
            return True
            
        # Check distance to each edge
        for i in range(3):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % 3]
            
            # Calculate distance from robot center to this edge
            dist = self._distance_point_to_segment(local_rx, local_ry, x1, y1, x2, y2)
            
            # Collision if distance is less than robot radius
            if dist < robot_radius:
                return True
                
        return False

    def intersects_segment(self, p1: tuple, p2: tuple, robot_radius: float, 
                         obs_x: float, obs_y: float) -> bool:
        """Checks if segment intersects triangle (with padding)."""
        # Transform segment endpoints to local coordinates
        p1_local = (p1[0] - obs_x, p1[1] - obs_y)
        p2_local = (p2[0] - obs_x, p2[1] - obs_y)
        
        # Check if endpoints are inside the triangle
        if self._point_in_triangle(*p1_local) or self._point_in_triangle(*p2_local):
            return True
            
        # Check for segment intersection with each triangle edge
        for i in range(3):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % 3]
            
            # Simple segment-segment intersection check
            if self._segments_intersect(p1_local, p2_local, v1, v2):
                return True
                
        # If no direct intersection, use sampling to check for robot radius clearance
        from_arr = np.array(p1)
        to_arr = np.array(p2)
        vec = to_arr - from_arr
        dist = np.linalg.norm(vec)
        
        if dist < 1e-9:  # Segment is essentially a point
            return self.check_collision(p1[0], p1[1], robot_radius, obs_x, obs_y)
            
        # Sample along the segment to check for collisions with radius
        num_steps = max(int(dist / (robot_radius * 0.5)) + 1, 2)
        
        for i in range(num_steps + 1):
            t = i / num_steps
            check_point = from_arr + vec * t
            if self.check_collision(check_point[0], check_point[1], robot_radius, obs_x, obs_y):
                return True
                
        return False

    def _segments_intersect(self, p1: tuple, p2: tuple, p3: tuple, p4: tuple) -> bool:
        """
        Check if segment p1-p2 intersects with segment p3-p4.
        Uses the cross product method.
        """
        def cross(p1, p2, p3):
            return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        
        def on_segment(p, q, r):
            return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                    min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
        ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

        # Xử lý các trường hợp đặc biệt (thẳng hàng)
        if d1 == 0 and on_segment(p3, p1, p4): return True
        if d2 == 0 and on_segment(p3, p2, p4): return True
        if d3 == 0 and on_segment(p1, p3, p2): return True
        if d4 == 0 and on_segment(p1, p4, p2): return True

        return False


    def get_patch(self, center_x: float, center_y: float, color: str, alpha: float, zorder: int) -> patches.Patch:
        # Transform triangle vertices to world coordinates
        world_vertices = [(v[0] + center_x, v[1] + center_y) for v in self.vertices]
        
        # Create and return a matplotlib Polygon patch
        return mplPolygon(world_vertices, closed=True, fc=color, alpha=alpha, zorder=zorder)

    def get_observation_params(self) -> tuple:
        """
        Format: (type, (x1, y1, x2, y2, x3, y3))
        """

        flat_vertices = self.vertices.flatten()

        # Return the type enum and first three parameters (remaining would need special handling)
        return (Triangle.SHAPE_TYPE_ENUM, tuple(flat_vertices))

    def get_efficient_distance(self, point_x: float, point_y: float, obs_x: float, obs_y: float) -> float:
        """Return the signed distance from a point to the triangle boundary."""
        # Convert to local coordinates
        local_x, local_y = self._transform_point(point_x, point_y, obs_x, obs_y)
        
        # Check if point is inside the triangle
        if self._point_in_triangle(local_x, local_y):
            # For points inside, compute distance to closest edge (negative)
            min_dist = float('inf')
            for i in range(3):
                x1, y1 = self.vertices[i]
                x2, y2 = self.vertices[(i + 1) % 3]
                dist = self._distance_point_to_segment(local_x, local_y, x1, y1, x2, y2)
                min_dist = min(min_dist, dist)
            return -min_dist
        else:
            # For points outside, compute distance to closest edge or vertex (positive)
            min_dist = float('inf')
            for i in range(3):
                x1, y1 = self.vertices[i]
                x2, y2 = self.vertices[(i + 1) % 3]
                dist = self._distance_point_to_segment(local_x, local_y, x1, y1, x2, y2)
                min_dist = min(min_dist, dist)
            return min_dist

    def get_centroid(self) -> np.ndarray:
        """Return the centroid of the triangle (always at the origin in local coords)."""
        # For a triangle, the centroid is at the arithmetic mean of its vertices
        centroid = np.mean(self.vertices, axis=0)
        return centroid

    def get_effective_radius(self) -> float:
        """Return the radius of the circumscribing circle of the triangle."""
        # Find the maximum distance from centroid to any vertex
        centroid = self.get_centroid()
        max_dist = 0.0
        for vertex in self.vertices:
            dist = np.linalg.norm(vertex - centroid)
            max_dist = max(max_dist, dist)
        return max_dist
    
    def get_effective_vector(self, point_x: float, point_y: float, obs_x: float, obs_y: float) -> np.ndarray:
        """
        Returns the shortest vector from point to the triangle's boundary.
        
        Args:
            point_x, point_y: Coordinates of the point
            obs_x, obs_y: Coordinates of the triangle's center/reference
            
        Returns:
            Vector from point to closest point on triangle boundary (as numpy array)
        """
        # Transform to local coordinates
        local_x, local_y = self._transform_point(point_x, point_y, obs_x, obs_y)
        local_point = (local_x, local_y)
        
        # If point is inside triangle, find closest edge
        if self._point_in_triangle(local_x, local_y):
            min_dist = float('inf')
            closest_point = None
            
            # Check distance to each edge
            for i in range(3):
                v1 = self.vertices[i]
                v2 = self.vertices[(i + 1) % 3]
                
                # Find closest point on this edge
                edge_closest_x, edge_closest_y = self._closest_point_on_segment(
                    local_x, local_y, v1[0], v1[1], v2[0], v2[1])
                
                dist = math.sqrt((local_x - edge_closest_x)**2 + (local_y - edge_closest_y)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_point = (edge_closest_x, edge_closest_y)
        else:
            # Point is outside, find closest point on any edge or vertex
            min_dist = float('inf')
            closest_point = None
            
            # Check each edge
            for i in range(3):
                v1 = self.vertices[i]
                v2 = self.vertices[(i + 1) % 3]
                
                # Find closest point on this edge
                edge_closest_x, edge_closest_y = self._closest_point_on_segment(
                    local_x, local_y, v1[0], v1[1], v2[0], v2[1])
                
                dist = math.sqrt((local_x - edge_closest_x)**2 + (local_y - edge_closest_y)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_point = (edge_closest_x, edge_closest_y)
        
        # Transform closest point back to world coordinates
        world_closest_x = closest_point[0] + obs_x
        world_closest_y = closest_point[1] + obs_y
        
        # Return vector from point to closest point
        return np.array([world_closest_x - point_x, world_closest_y - point_y])
    
    # def intersect_ray(self, ray_origin: np.ndarray, ray_direction: np.ndarray, obs_x: float, obs_y: float) -> float:
    #     """
    #     Tính khoảng cách giao của một tia với hình tam giác này.

    #     Args:
    #         ray_origin: np.ndarray (x, y), điểm bắt đầu của tia (tọa độ thế giới).
    #         ray_direction: np.ndarray (dx, dy), vector hướng đơn vị của tia (tọa độ thế giới).
    #         obs_x: Tọa độ x điểm tham chiếu của vật cản (tọa độ thế giới).
    #         obs_y: Tọa độ y điểm tham chiếu của vật cản (tọa độ thế giới).

    #     Returns:
    #         float: Khoảng cách 't' dọc theo hướng tia nơi giao cắt *đầu tiên*
    #                xảy ra (P_intersect = ray_origin + t * ray_direction).
    #                Trả về float('inf') nếu không có giao cắt.
    #     """
    #     # Chuyển đổi tia sang hệ tọa độ cục bộ của tam giác
    #     local_ray_origin = ray_origin - np.array([obs_x, obs_y])
    #     # local_ray_direction không đổi vì chỉ tịnh tiến

    #     min_t = float('inf')
    #     epsilon = 1e-9 # Giá trị nhỏ cho so sánh dấu phẩy động

    #     # Kiểm tra giao cắt với từng cạnh của tam giác
    #     for i in range(3):
    #         p1 = self.vertices[i]          # Điểm bắt đầu của cạnh (local)
    #         p2 = self.vertices[(i + 1) % 3] # Điểm kết thúc của cạnh (local)

    #         edge_vec = p2 - p1             # Vector cạnh (vAB trong giải thích)
    #         origin_to_p1 = p1 - local_ray_origin # Vector từ gốc tia đến điểm đầu cạnh (vOA)

    #         # Tính định thức (cross product 2D: ray_direction x edge_vec)
    #         # det = Dx * vABy - Dy * vABx
    #         determinant = ray_direction[0] * edge_vec[1] - ray_direction[1] * edge_vec[0]

    #         # Nếu định thức gần bằng 0, tia song song với cạnh
    #         if abs(determinant) < epsilon:
    #             # Xử lý trường hợp thẳng hàng (nếu cần thiết, phức tạp hơn)
    #             # Hiện tại, bỏ qua giao cắt nếu song song để đơn giản hóa
    #             continue

    #         # Tính tham số u (vị trí trên đoạn thẳng cạnh) và t (khoảng cách dọc tia)
    #         # u = (Dx * vOAy - Dy * vOAx) / det = (ray_direction x origin_to_p1) / det
    #         u = (ray_direction[0] * origin_to_p1[1] - ray_direction[1] * origin_to_p1[0]) / determinant
    #         # t = (vOAx * vABy - vOAy * vABx) / det = (origin_to_p1 x edge_vec) / det
    #         t = (origin_to_p1[0] * edge_vec[1] - origin_to_p1[1] * edge_vec[0]) / determinant

    #         # Kiểm tra xem điểm giao có hợp lệ không
    #         # t >= 0: Giao cắt nằm phía trước hoặc tại gốc tia
    #         # 0 <= u <= 1: Giao cắt nằm trên đoạn thẳng của cạnh
    #         if t >= -epsilon and -epsilon <= u <= 1 + epsilon:
    #             min_t = min(min_t, t) # Cập nhật khoảng cách giao gần nhất

    #     # Nếu gốc tia nằm bên trong tam giác, khoảng cách giao là 0
    #     # if self._point_in_triangle(local_ray_origin[0], local_ray_origin[1]):
    #     #     return 0.0 # Trả về 0 nếu gốc tia ở bên trong (tuỳ chọn, có thể gây vấn đề nếu cần giao điểm thoát ra)

    #     # Đảm bảo chỉ trả về khoảng cách dương thực sự
    #     if min_t < epsilon:
    #          # Nếu min_t rất gần 0 (có thể do gốc tia nằm trên biên),
    #          # hãy xem xét nó là không giao cắt từ bên ngoài hoặc trả về một giá trị dương nhỏ
    #          # để tránh các vấn đề về chia cho 0 hoặc khoảng cách âm.
    #          # Tuy nhiên, nếu gốc tia nằm bên trong, logic ray-segment sẽ tìm thấy điểm thoát ra.
    #          # Đối với tránh va chạm, thường chỉ quan tâm đến t > 0.
    #          if self._point_in_triangle(local_ray_origin[0], local_ray_origin[1]):
    #               # Gốc tia bên trong, min_t tìm được là điểm thoát ra.
    #               # Nếu chỉ cần biết có va chạm không, có thể trả về 0.
    #               # Nếu cần điểm giao thực sự, min_t là hợp lệ.
    #               # Trả về min_t nếu nó dương, nếu không thì inf (không có điểm thoát hợp lệ?)
    #               return min_t if min_t > epsilon else float('inf') # Chỉ trả về t dương thực sự
    #          else:
    #               # Gốc tia bên ngoài, nhưng t gần 0 (trên biên). Coi như không giao.
    #               # Hoặc trả về 0? Trả về inf để rõ ràng là không có giao cắt "phía trước".
    #               return float('inf')
    #     return min_t

    def intersect_ray(self, ray_origin: np.ndarray, ray_direction: np.ndarray, obs_x: float, obs_y: float) -> float:
        """
        Calculates the intersection distance of a ray with this triangle instance.

        Args:
            ray_origin: np.ndarray (x, y), starting point of the ray (world coords).
            ray_direction: np.ndarray (dx, dy), unit direction vector of the ray (world coords).
            obs_x: Obstacle's reference point x-coordinate (world).
            obs_y: Obstacle's reference point y-coordinate (world).

        Returns:
            float: The distance 't' along the ray direction where the *first* intersection
                   occurs (P_intersect = ray_origin + t * ray_direction).
                   Returns float('inf') if there is no intersection with t >= 0.
        """
        # Transform ray to triangle's local coordinate system
        local_ray_origin = ray_origin - np.array([obs_x, obs_y])
        # Direction remains the same under translation
        local_ray_dir = ray_direction

        min_t = float('inf')
        epsilon = 1e-9 # Epsilon for floating point comparisons

        # Check intersection with each of the 3 edges (segments) of the triangle
        for i in range(3):
            # Get edge vertices in local coordinates
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % 3]

            # Edge direction vector
            edge_dir = p2 - p1

            # --- Solve for intersection using determinant method ---
            # We want to find t, s such that:
            # local_ray_origin + t * local_ray_dir = p1 + s * edge_dir
            # Rearranged: t * local_ray_dir - s * edge_dir = p1 - local_ray_origin

            # Matrix form: [[dir_x, -edge_x], [dir_y, -edge_y]] * [[t], [s]] = [[p1x-origx], [p1y-origy]]
            matrix_det = np.cross(local_ray_dir, edge_dir) # dir_x*(-edge_y) - dir_y*(-edge_x)

            # Check if ray is parallel to the edge segment
            if abs(matrix_det) < epsilon:
                # Parallel: Check for collinearity and overlap (optional, often ignored for robustness)
                # If collinear, an intersection might occur if the ray origin lies on the infinite line
                # and points towards the segment, or if the origin is within the segment.
                # Simplified: Treat parallel as no intersection for now.
                continue # Skip to next edge

            # Vector from ray origin to edge start point (p1)
            origin_to_p1 = p1 - local_ray_origin

            # Solve for t and s using Cramer's rule or properties of cross product
            # t = (origin_to_p1 x edge_dir) / (local_ray_dir x edge_dir)
            t = np.cross(origin_to_p1, edge_dir) / matrix_det
            # s = (origin_to_p1 x local_ray_dir) / (local_ray_dir x edge_dir)
            s = np.cross(origin_to_p1, local_ray_dir) / matrix_det

            # --- Validate Intersection ---
            # 1. Intersection must be forward along the ray (t >= 0)
            # 2. Intersection must be within the segment bounds (0 <= s <= 1)
            if t >= -epsilon and (s >= -epsilon and s <= 1.0 + epsilon):
                # Valid intersection found
                min_t = min(min_t, t)


        # Ensure the returned t is non-negative
        if min_t == float('inf'):
            return float('inf') # No valid intersection found
        else:
            return max(0.0, min_t) # Clamp potential small negative t due to epsilon to 0

class Polygon(Shape):
    """
    Polygon Shape. Implements a general polygon using triangulation.
    """
    SHAPE_TYPE_ENUM = 3

    def __init__(self, vertices: List[Tuple[float, float]]):
        """
        Initialize a polygon from a list of vertices.
        
        Args:
            vertices: List of (x, y) coordinates defining the polygon vertices
                     in the local coordinate system (centered at origin)
        """
        if len(vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        
        # Convert to numpy array for easier manipulation
        self.vertices = np.array(vertices)
        
        # Sort vertices clockwise
        self._sort_vertices_clockwise()
        
        # Perform triangulation of the polygon
        self.triangles = self._triangulate()
        
        # Calculate centroid
        self._centroid = self._calculate_centroid()
        
        # Calculate effective radius
        self._effective_radius = self._calculate_effective_radius()

    def _sort_vertices_clockwise(self):
        """Sort the vertices in clockwise order."""
        # Calculate centroid
        centroid = np.mean(self.vertices, axis=0)
        
        # Sort vertices based on angle around centroid
        def get_angle(vertex):
            return -math.atan2(vertex[1] - centroid[1], vertex[0] - centroid[0])
        
        # Sort vertices by angle
        sorted_vertices = sorted(self.vertices, key=get_angle)
        self.vertices = np.array(sorted_vertices)

    def _triangulate(self) -> List[Triangle]:
        """
        Triangulate the polygon using ear clipping algorithm.
        Returns a list of Triangle objects.
        """
        triangles = []
        
        # Make a copy of vertices that we can modify
        remaining_vertices = list(self.vertices)
        
        # Continue until we have triangulated the entire polygon
        while len(remaining_vertices) > 3:
            # Find an ear to clip
            ear_index = self._find_ear(remaining_vertices)
            
            if ear_index == -1:
                # Should not happen for simple polygons
                raise ValueError("Failed to triangulate polygon. Check if it's simple and non-degenerate.")
            
            # Create triangle from ear
            i_prev = (ear_index - 1) % len(remaining_vertices)
            i_next = (ear_index + 1) % len(remaining_vertices)
            
            triangle_vertices = [
                remaining_vertices[i_prev],
                remaining_vertices[ear_index],
                remaining_vertices[i_next]
            ]
            
            triangles.append(Triangle(triangle_vertices))
            
            # Remove ear vertex
            remaining_vertices.pop(ear_index)
        
        # Add the final triangle (remaining 3 vertices)
        if len(remaining_vertices) == 3:
            triangles.append(Triangle(remaining_vertices))
            
        return triangles

    def _find_ear(self, vertices: List) -> int:
        """
        Find an ear vertex in the polygon.
        An ear is a vertex where the triangle formed with its adjacent vertices
        contains no other vertices.
        
        Args:
            vertices: List of remaining vertices
            
        Returns:
            Index of an ear vertex, or -1 if none found
        """
        n = len(vertices)
        
        # Check each vertex
        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            
            v0 = vertices[prev_idx]
            v1 = vertices[i]
            v2 = vertices[next_idx]
            
            # Check if the vertex forms a convex angle (required for an ear)
            if not self._is_convex(v0, v1, v2):
                continue
                
            # Check if the triangle contains any other vertices
            contains_other_vertex = False
            for j in range(n):
                if j == prev_idx or j == i or j == next_idx:
                    continue
                    
                if self._point_in_triangle(vertices[j], v0, v1, v2):
                    contains_other_vertex = True
                    break
                    
            if not contains_other_vertex:
                return i
                
        return -1  # No ear found (should not happen for simple polygons)

    def _is_convex(self, v0, v1, v2) -> bool:
        """
        Check if three consecutive vertices form a convex angle.
        
        Args:
            v0, v1, v2: Three consecutive vertices
            
        Returns:
            True if the angle is convex, False otherwise
        """
        # Cross product to determine if v1 is a convex vertex
        cross_product = (v1[0] - v0[0]) * (v2[1] - v1[1]) - (v1[1] - v0[1]) * (v2[0] - v1[0])
        
        # If cross product is negative, the angle is convex (for clockwise vertices)
        return cross_product < 0

    def _point_in_triangle(self, p, v0, v1, v2) -> bool:
        """
        Check if a point is inside a triangle.
        Uses barycentric coordinates.
        
        Args:
            p: Point to check
            v0, v1, v2: Triangle vertices
        
        Returns:
            True if point is inside the triangle, False otherwise
        """
        # Convert to numpy arrays for easier calculation
        p = np.array(p)
        v0 = np.array(v0)
        v1 = np.array(v1)
        v2 = np.array(v2)
        
        # Compute barycentric coordinates
        area = 0.5 * np.abs(np.cross(v1 - v0, v2 - v0))
        
        # If triangle has zero area, point can't be inside
        if area < 1e-10:
            return False
            
        alpha = np.abs(np.cross(p - v1, v2 - v1)) / (2 * area)
        beta = np.abs(np.cross(p - v2, v0 - v2)) / (2 * area)
        gamma = np.abs(np.cross(p - v0, v1 - v0)) / (2 * area)
        
        # Check if sum of barycentric coordinates is approximately 1
        return abs(alpha + beta + gamma - 1.0) < 1e-9

    def _calculate_centroid(self) -> np.ndarray:
        """Calculate and return the centroid of the polygon."""
        # For a polygon, area-weighted average of vertices
        n = len(self.vertices)
        area = 0.0
        centroid_x = 0.0
        centroid_y = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            cross = self.vertices[i][0] * self.vertices[j][1] - self.vertices[j][0] * self.vertices[i][1]
            area += cross
            centroid_x += (self.vertices[i][0] + self.vertices[j][0]) * cross
            centroid_y += (self.vertices[i][1] + self.vertices[j][1]) * cross
            
        area *= 0.5
        centroid_x /= (6.0 * area)
        centroid_y /= (6.0 * area)
        
        return np.array([centroid_x, centroid_y])

    def _calculate_effective_radius(self) -> float:
        """Calculate the effective radius (maximum distance from centroid to any vertex)."""
        max_dist = 0.0
        for vertex in self.vertices:
            dist = np.linalg.norm(vertex - self._centroid)
            max_dist = max(max_dist, dist)
        return max_dist

    def check_collision(self, robot_x: float, robot_y: float, robot_radius: float, 
                        obs_x: float, obs_y: float) -> bool:
        """
        Check collision between robot circle and polygon.
        Uses component triangles for the check.
        """
        # Transform robot center to local coordinates
        local_rx = robot_x - obs_x
        local_ry = robot_y - obs_y
        
        # Check collision with any component triangle
        for triangle in self.triangles:
            if triangle.check_collision(local_rx, local_ry, robot_radius, 0, 0):
                return True
                
        return False

    def intersects_segment(self, p1: tuple, p2: tuple, robot_radius: float, 
                         obs_x: float, obs_y: float) -> bool:
        """
        Check if segment intersects polygon (with padding).
        Uses component triangles for the check.
        """
        # Transform segment endpoints to local coordinates
        p1_local = (p1[0] - obs_x, p1[1] - obs_y)
        p2_local = (p2[0] - obs_x, p2[1] - obs_y)
        
        # Check intersection with any component triangle
        for triangle in self.triangles:
            if triangle.intersects_segment(p1_local, p2_local, robot_radius, 0, 0):
                return True
                
        return False

    def get_patch(self, center_x: float, center_y: float, color: str, alpha: float, zorder: int) -> patches.Patch:
        """Create a matplotlib patch for the polygon."""
        # Transform polygon vertices to world coordinates
        world_vertices = [(v[0] + center_x, v[1] + center_y) for v in self.vertices]
        
        # Create and return a matplotlib Polygon patch
        return mplPolygon(world_vertices, closed=True, fc=color, alpha=alpha, zorder=zorder)

    def get_observation_params(self) -> tuple:
        """
        Format: (type, (x1, y1, x2, y2, ..., xn, yn))
        Returns the shape type and flattened vertices.
        """
        flat_vertices = self.vertices.flatten()
        return (Polygon.SHAPE_TYPE_ENUM, tuple(flat_vertices))

    def get_efficient_distance(self, point_x: float, point_y: float, obs_x: float, obs_y: float) -> float:
        """
        Calculate the signed distance from point to polygon boundary.
        Uses component triangles and calculates the minimum.
        """
        # Transform to local coordinates
        local_x = point_x - obs_x
        local_y = point_y - obs_y
        
        # Calculate minimum distance to any triangle
        min_distance = float('inf')
        inside_any = False
        
        for triangle in self.triangles:
            dist = triangle.get_efficient_distance(local_x, local_y, 0, 0)
            
            # If inside any triangle, we're inside the polygon
            if dist < 0:
                inside_any = True
                min_distance = min(min_distance, abs(dist))
            elif not inside_any:
                # Only update outside distance if we haven't found an inside case yet
                min_distance = min(min_distance, dist)
        
        # Return negative distance if inside the polygon
        return -min_distance if inside_any else min_distance

    def get_centroid(self) -> np.ndarray:
        """Return the pre-calculated centroid of the polygon."""
        return self._centroid

    def get_effective_radius(self) -> float:
        """Return the pre-calculated effective radius of the polygon."""
        return self._effective_radius
    
    def get_effective_vector(self, point_x: float, point_y: float, obs_x: float, obs_y: float) -> np.ndarray:
        """
        Returns the shortest vector from point to the polygon's boundary.
        
        Args:
            point_x, point_y: Coordinates of the point
            obs_x, obs_y: Coordinates of the polygon's reference point
            
        Returns:
            Vector from point to closest point on polygon boundary (as numpy array)
        """
        # Transform to local coordinates
        local_x = point_x - obs_x
        local_y = point_y - obs_y
        
        # Check if point is inside any component triangle
        inside_polygon = False
        for triangle in self.triangles:
            # Use local coordinates (offset from triangle is already 0,0)
            if triangle._point_in_triangle(local_x, local_y):
                inside_polygon = True
                break
        
        # Initialize variables for finding closest point
        min_dist = float('inf')
        closest_point = None
        
        if inside_polygon:
            # If inside, find closest edge of the polygon
            for i in range(len(self.vertices)):
                v1 = self.vertices[i]
                v2 = self.vertices[(i + 1) % len(self.vertices)]
                
                # Find closest point on this edge
                edge_closest_x, edge_closest_y = self._closest_point_on_segment(
                    local_x, local_y, v1[0], v1[1], v2[0], v2[1])
                
                dist = math.sqrt((local_x - edge_closest_x)**2 + (local_y - edge_closest_y)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_point = (edge_closest_x, edge_closest_y)
        else:
            # If outside, find closest edge or vertex of the polygon
            for i in range(len(self.vertices)):
                v1 = self.vertices[i]
                v2 = self.vertices[(i + 1) % len(self.vertices)]
                
                # Find closest point on this edge
                edge_closest_x, edge_closest_y = self._closest_point_on_segment(
                    local_x, local_y, v1[0], v1[1], v2[0], v2[1])
                
                dist = math.sqrt((local_x - edge_closest_x)**2 + (local_y - edge_closest_y)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_point = (edge_closest_x, edge_closest_y)
        
        # Transform closest point back to world coordinates
        world_closest_x = closest_point[0] + obs_x
        world_closest_y = closest_point[1] + obs_y
        
        # Return vector from point to closest point
        return np.array([world_closest_x - point_x, world_closest_y - point_y])

    # Polygon needs a helper method for finding closest point on segment
    def _closest_point_on_segment(self, px, py, ax, ay, bx, by):
        """
        Find closest point on line segment (a-b) to point p.
        
        Args:
            px, py: Point coordinates
            ax, ay: First segment endpoint
            bx, by: Second segment endpoint
            
        Returns:
            Tuple of coordinates for closest point on segment
        """
        # Vector from a to b
        abx = bx - ax
        aby = by - ay
        
        # Square of length of AB
        ab_sq_len = abx * abx + aby * aby
        
        # If segment is a point, return a
        if ab_sq_len < 1e-10:
            return ax, ay
        
        # Project p onto ab, computing the normalized parameter
        ap_dot_ab = (px - ax) * abx + (py - ay) * aby
        t = ap_dot_ab / ab_sq_len
        
        # Clamp t to [0,1] to ensure point is on segment
        t = max(0, min(1, t))
        
        # Return point coordinates
        return ax + t * abx, ay + t * aby
    
    def intersect_ray(self, ray_origin: np.ndarray, ray_direction: np.ndarray, obs_x: float, obs_y: float) -> float:
        """
        Calculates the intersection distance of a ray with this polygon instance.
        Checks intersection against all component triangles.

        Args:
            ray_origin: np.ndarray (x, y), starting point of the ray (world coords).
            ray_direction: np.ndarray (dx, dy), unit direction vector of the ray (world coords).
            obs_x: Obstacle's reference point x-coordinate (world).
            obs_y: Obstacle's reference point y-coordinate (world).

        Returns:
            float: The distance 't' along the ray direction where the *first* intersection
                   occurs (P_intersect = ray_origin + t * ray_direction).
                   Returns float('inf') if there is no intersection with t >= 0.
        """
        min_t = float('inf')

        # Iterate through each component triangle
        for triangle in self.triangles:
            # Call the triangle's intersect_ray method.
            # Since triangle vertices are already relative to the polygon's origin (obs_x, obs_y),
            # we can pass the world ray and polygon origin directly, OR transform the ray
            # into the polygon's local system and call triangle.intersect_ray with (0,0) offset.
            # Let's use the latter approach for consistency with intersects_segment.

            # Transform ray to polygon's local coordinate system (only origin shifts)
            local_ray_origin = ray_origin - np.array([obs_x, obs_y])
            # Direction vector remains the same

            # Call triangle's intersect_ray with local ray and (0,0) offset
            t = triangle.intersect_ray(local_ray_origin, ray_direction, 0.0, 0.0)

            # Update minimum intersection distance found so far
            if t < min_t:
                 min_t = t

        return min_t # Returns float('inf') if no triangle was hit