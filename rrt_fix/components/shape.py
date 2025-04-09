import math
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.patches as patches
from typing import List, Tuple
from matplotlib import pyplot as plt
from matplotlib.patches import Circle as mplCircle
from matplotlib.patches import Rectangle as mplRectangle
from matplotlib.patches import Polygon as mplPolygon

# --- Shape Definitions ---

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
            obs_x: Obstacle's reference point x-coordinate (e.g., center).
            obs_y: Obstacle's reference point y-coordinate (e.g., center).

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
        Format: (shape_type_enum, param1, param2, param3)
        shape_type_enum: 0=Circle, 1=Rectangle, ...
        param1, param2, param3: Shape-specific dimensions (e.g., radius, width, height, angle).
                                Use 0 or NaN for unused parameters.
        """
        pass

    @abstractmethod
    def get_efficient_distance(self, point_x: float, point_y: float, obs_x: float, obs_y: float) -> float:
        """
        Returns the distance from the shape to a point (point_x, point_y).
        This is the distance from point to boundary of shape.

        Args:
            point_x: X-coordinate of the point.
            point_y: Y-coordinate of the point.
            obs_x: Obstacle's reference point x-coordinate.
            obs_y: Obstacle's reference point y-coordinate.

        Returns:
            Distance from the shape to the point.
        """
        pass

    @abstractmethod
    def get_centroid(self) -> np.ndarray:
        """ Returns the centroid relative to the shape's origin (usually 0,0). """
        pass # Might not be needed if obs_x, obs_y is always the centroid


class Circle(Shape):
    """Circular Shape."""
    SHAPE_TYPE_ENUM = 0

    def __init__(self, radius: float):
        if radius <= 0:
            raise ValueError("Circle radius must be positive")
        self.radius = radius

    def check_collision(self, robot_x: float, robot_y: float, robot_radius: float, obs_x: float, obs_y: float) -> bool:
        """Checks collision between robot circle and obstacle circle."""
        min_dist = self.radius + robot_radius
        dist_sq = (robot_x - obs_x)**2 + (robot_y - obs_y)**2
        return dist_sq < min_dist**2

    def intersects_segment(self, p1: tuple, p2: tuple, robot_radius: float, obs_x: float, obs_y: float) -> bool:
        """Checks if segment intersects circle (with padding)."""
        # Check distance from circle center (obs_x, obs_y) to the line segment p1-p2.
        # Include robot_radius as padding for the check distance.
        effective_radius = self.radius + robot_radius
        p1_arr = np.array(p1)
        p2_arr = np.array(p2)
        obs_pos = np.array([obs_x, obs_y])
        segment_vec = p2_arr - p1_arr
        segment_len_sq = np.dot(segment_vec, segment_vec)

        # If segment is a point
        if segment_len_sq < 1e-12:
            return np.linalg.norm(p1_arr - obs_pos) < effective_radius

        # Project obstacle center onto the line defined by the segment
        # t = dot(obs_pos - p1, p2 - p1) / |p2 - p1|^2
        t = np.dot(obs_pos - p1_arr, segment_vec) / segment_len_sq

        # Find the closest point on the *line* to the obstacle center
        if t < 0.0:
            closest_point_on_line = p1_arr
        elif t > 1.0:
            closest_point_on_line = p2_arr
        else:
            closest_point_on_line = p1_arr + t * segment_vec

        # Check distance from obstacle center to this closest point
        dist_sq_to_segment = np.sum((obs_pos - closest_point_on_line)**2)

        return dist_sq_to_segment < effective_radius**2

    def get_patch(self, center_x: float, center_y: float, color: str, alpha: float, zorder: int) -> patches.Patch:
        return patches.Circle((center_x, center_y), self.radius, fc=color, alpha=alpha, zorder=zorder)

    def get_observation_params(self) -> tuple:
        # type, radius, unused, unused
        return (Circle.SHAPE_TYPE_ENUM, self.radius, 0.0, 0.0)
    
    def get_efficient_distance(self, point_x, point_y, obs_x, obs_y):
        return np.sqrt((point_x - obs_x)**2 + (point_y - obs_y)**2) - self.radius

    def get_centroid(self) -> np.ndarray:
        return np.array([0.0, 0.0])


class Rectangle(Shape):
    """Rectangular Shape, potentially rotated."""
    SHAPE_TYPE_ENUM = 1

    def __init__(self, width: float, height: float, angle: float = 0.0):
        """
        Args:
            width: Width along the rectangle's local x-axis.
            height: Height along the rectangle's local y-axis.
            angle: Rotation angle in radians relative to the world x-axis.
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

    def _world_to_local(self, point_x, point_y, obs_x, obs_y):
        """Transforms world coordinates to the rectangle's local frame."""
        dx = point_x - obs_x
        dy = point_y - obs_y
        local_x = dx * self._cos_a + dy * self._sin_a
        local_y = -dx * self._sin_a + dy * self._cos_a
        return local_x, local_y

    def _get_corners(self, obs_x, obs_y):
        """ Get world coordinates of the rectangle corners. """
        # Local corner offsets
        corners_local = [
            (self._half_w, self._half_h),
            (-self._half_w, self._half_h),
            (-self._half_w, -self._half_h),
            (self._half_w, -self._half_h)
        ]
        # Rotate and translate
        corners_world = []
        for lx, ly in corners_local:
            wx = obs_x + lx * self._cos_a - ly * self._sin_a
            wy = obs_y + lx * self._sin_a + ly * self._cos_a
            corners_world.append((wx, wy))
        return corners_world

    def check_collision(self, robot_x: float, robot_y: float, robot_radius: float, obs_x: float, obs_y: float) -> bool:
        """Checks collision between robot circle and obstacle rectangle."""
        # This is complex due to rounded corners (Minkowski sum).
        # Approximation 1: Check if robot center is inside rectangle expanded by robot_radius.
        # Approximation 2: Check distance from robot center to the rectangle shape.
        # Approximation 3 (Easier): Check if the distance from the robot center to the *closest point*
        # on the rectangle's perimeter is less than the robot_radius.

        # Transform robot center to rectangle's local coordinates
        local_rx, local_ry = self._world_to_local(robot_x, robot_y, obs_x, obs_y)

        # Find closest point on the infinite line boundaries
        clamped_x = max(-self._half_w, min(self._half_w, local_rx))
        clamped_y = max(-self._half_h, min(self._half_h, local_ry))

        # The closest point on the rectangle to the local robot pos
        closest_local_x = local_rx
        closest_local_y = local_ry

        # If robot is outside the x-bounds
        if local_rx < -self._half_w or local_rx > self._half_w:
            closest_local_x = clamped_x
        # If robot is outside the y-bounds
        if local_ry < -self._half_h or local_ry > self._half_h:
            closest_local_y = clamped_y
        # If robot is inside in one dim but outside in another,
        # the closest point might be a corner - handled by clamping above.
        # If inside both, the closest point is the robot center itself (local_rx, local_ry)
        # and the distance is 0. But we need the closest point *on the perimeter* if inside.
        # So, if inside, calculate distance to each edge and find minimum?
        # Let's use the clamped point approach - it finds the closest point on the solid rectangle.

        closest_point_on_rect = np.array([closest_local_x, closest_local_y])
        local_robot_pos = np.array([local_rx, local_ry])

        dist_sq = np.sum((local_robot_pos - closest_point_on_rect)**2)

        return dist_sq < robot_radius**2

    def intersects_segment(self, p1: tuple, p2: tuple, robot_radius: float, obs_x: float, obs_y: float) -> bool:
        """Checks if segment intersects rectangle (with padding)."""
        # This is complex with padding. Simplification:
        # Use the RRT approach: sample points along the segment and check collision
        # for each point using the shape's check_collision method.

        from_arr = np.array(p1)
        to_arr = np.array(p2)
        vec = to_arr - from_arr
        dist = np.linalg.norm(vec)

        if dist < 1e-6: # Segment is a point
             return self.check_collision(p1[0], p1[1], robot_radius, obs_x, obs_y)

        unit_vec = vec / dist
        # Check intermediate points frequently
        num_checks = max(int(dist / (robot_radius * 0.5)) + 1, 2) # Check more often

        for i in range(num_checks + 1): # Check points including start and end
            t = i / num_checks
            check_point = from_arr + unit_vec * (dist * t)
            # Check collision of this point (as a robot center) with the rectangle
            if self.check_collision(check_point[0], check_point[1], robot_radius, obs_x, obs_y):
                 return True # Collision detected for this point on segment
        return False # No collision found for any checked point

    def get_patch(self, center_x: float, center_y: float, color: str, alpha: float, zorder: int) -> patches.Patch:
        # Rectangle patch takes bottom-left corner, width, height, angle
        # Calculate bottom-left corner in world coordinates
        bl_local_x = -self._half_w
        bl_local_y = -self._half_h
        # Rotate and translate local bottom-left to world
        bl_world_x = center_x + bl_local_x * self._cos_a - bl_local_y * self._sin_a
        bl_world_y = center_y + bl_local_x * self._sin_a + bl_local_y * self._cos_a

        return patches.Rectangle((bl_world_x, bl_world_y), self.width, self.height,
                                 angle=math.degrees(self.angle), # Patches uses degrees
                                 fc=color, alpha=alpha, zorder=zorder)

    def get_observation_params(self) -> tuple:
        # type, width, height, angle
        return (Rectangle.SHAPE_TYPE_ENUM, self.width, self.height, self.angle)
    
    def get_efficient_distance(self, point_x, point_y, obs_x, obs_y):
        # Transform point to local coordinates
        local_x, local_y = self._world_to_local(point_x, point_y, obs_x, obs_y)
        # Calculate distance to the rectangle's edges
        clamped_x = max(-self._half_w, min(self._half_w, local_x))
        clamped_y = max(-self._half_h, min(self._half_h, local_y))
        closest_point_on_rect = np.array([clamped_x, clamped_y])
        local_point_pos = np.array([local_x, local_y])
        dist_sq = np.sum((local_point_pos - closest_point_on_rect)**2)
        return math.sqrt(dist_sq)

    def get_centroid(self) -> np.ndarray:
        return np.array([0.0, 0.0]) # Centroid is at the reference point (obs_x, obs_y)
