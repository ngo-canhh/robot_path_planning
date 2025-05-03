# --- START OF FILE obstacle.py ---

import math
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import matplotlib.patches as patches
from components.shape import Shape

# --- Obstacle Definitions ---

class ObstacleType(Enum):
    STATIC = 0
    DYNAMIC = 1

class Obstacle(ABC):
    """Abstract base class for obstacles."""

    def __init__(self, x: float, y: float, shape: Shape, obs_type: ObstacleType, velocity: float = 0, direction: np.ndarray = None):
        self.x = x
        self.y = y
        self.shape = shape
        self.type = obs_type
        self.velocity = velocity if velocity is not None else 0
        # Ensure direction is a numpy array and normalized for dynamic obstacles
        if direction is not None:
            direction = np.array(direction, dtype=float)
            norm = np.linalg.norm(direction)
            if norm > 1e-6: # Avoid division by zero
                 self.direction = direction / norm
            else:
                 self.direction = np.array([0.0, 0.0]) # Default direction if norm is zero
        else:
            self.direction = np.array([0.0, 0.0]) # Default direction if None

        # Store initial state for potential reset (e.g., for planner map)
        self.initial_x = x
        self.initial_y = y
        # Keep initial direction/velocity too if they can change? For now, assume they are constant or handled by update.

    @abstractmethod
    def update(self, dt: float = 1.0, bounds: tuple = (0, 0, 500, 500)):
        """Updates the obstacle's state (e.g., position for dynamic obstacles)."""
        pass

    def get_position(self) -> np.ndarray:
        """Returns the current reference position (e.g., center)."""
        return np.array([self.x, self.y])

    def check_collision(self, robot_x: float, robot_y: float, robot_radius: float) -> bool:
        """Checks collision between the robot and this obstacle."""
        # Delegate to the shape's collision check
        return self.shape.check_collision(robot_x, robot_y, robot_radius, self.x, self.y)

    def intersects_segment(self, p1: tuple, p2: tuple, robot_radius: float) -> bool:
        """Checks if a padded segment intersects this obstacle."""
        # Delegate to the shape's intersection check
        return self.shape.intersects_segment(p1, p2, robot_radius, self.x, self.y)

    def get_render_patch(self, color: str = 'gray', alpha: float = 0.6, zorder: int = 3) -> patches.Patch:
        """Gets the matplotlib patch for rendering."""
        # Delegate to the shape's patch creation
        return self.shape.get_patch(self.x, self.y, color=color, alpha=alpha, zorder=zorder)

    def get_observation_data(self) -> dict: # Changed return type hint to dict
        """
        Returns obstacle data formatted for the environment's observation space.
        Format: {
            'x': float - obs_x in world coordinate,
            'y': float - obs_y in world coordinate,
            'shape_type': int - shape type enum,
            'shape_params': tuple - params of the shape, depend on shape type (eg. 'radius' for Circle, 'width', 'height', 'angle' for Rectangle, etc.),
            'dynamic_flag': bool - 1 for dynamic, 0 for static,
            'vel_x': float - velocity x-axis of obs,
            'vel_y': velocity y-axis of obs,
            'bounding_box': tuple - (x_min, y_min, x_max, y_max) for bounding box if applicable

        }
        """
        pass

    def reset_to_initial(self):
         """Resets obstacle to its starting position (useful for planner map)."""
         self.x = self.initial_x
         self.y = self.initial_y
         # Should velocity/direction also be reset if they could change? Assume not for now.

    def get_efficient_distance(self, point_x: float, point_y: float):
        return self.shape.get_efficient_distance(point_x, point_y, self.x, self.y)
    
    # Added for better comparison in memory management
    def get_static_description(self) -> tuple:
        """Returns a tuple describing the static properties (position, shape) for comparison."""
        shape_type, shape_params = self.shape.get_observation_params()
        # Round position to handle potential float inaccuracies if needed, 
        # but exact initial position is usually better for comparison.
        return (self.initial_x, self.initial_y, shape_type, shape_params)


class StaticObstacle(Obstacle):
    """Obstacle that does not move."""
    def __init__(self, x: float, y: float, shape: Shape):
        super().__init__(x, y, shape, ObstacleType.STATIC)

    def update(self, dt: float = 1.0, bounds: tuple = (0, 0, 500, 500)):
        # Static obstacles don't move
        pass

    def get_observation_data(self) -> dict:
        shape_type, shape_params = self.shape.get_observation_params()
        dynamic_flag = 1.0 if self.type == ObstacleType.DYNAMIC else 0.0
        vel_x = self.velocity * self.direction[0] if self.type == ObstacleType.DYNAMIC else 0.0
        vel_y = self.velocity * self.direction[1] if self.type == ObstacleType.DYNAMIC else 0.0
        return {
            'x': self.x,
            'y': self.y,
            'shape_type': shape_type,
            'shape_params': shape_params,
            'dynamic_flag': dynamic_flag,
            'vel_x': vel_x,
            'vel_y': vel_y,
            'bounding_box': None # Static obstacles don't have a bounding box
        }

    def get_render_patch(self, alpha: float = 0.6, zorder: int = 3) -> patches.Patch:
        return super().get_render_patch(color='dimgray', alpha=alpha, zorder=zorder)


class DynamicObstacle(Obstacle):
    """
    Obstacle that moves with a constant velocity.
    If bounding_box is provided, it bounces within that region.
    Otherwise, it bounces off the global environment boundaries ('bounds').
    """
    def __init__(self, x: float, y: float, shape: Shape, velocity: float, direction: np.ndarray, bounding_box: tuple = None):
        """
        Initializes a DynamicObstacle.

        Args:
            x (float): Initial x-coordinate.
            y (float): Initial y-coordinate.
            shape (Shape): The shape of the obstacle.
            velocity (float): The speed of the obstacle.
            direction (np.ndarray): The initial direction vector (will be normalized).
            bounding_box (tuple, optional): Defines a rectangular region for movement
                                           (min_x, min_y, max_x, max_y). If None, uses
                                           global bounds from the update method. Defaults to None.
        """
        super().__init__(x, y, shape, ObstacleType.DYNAMIC, velocity, direction)
        self.bounding_box = bounding_box # Store the specific bounding box

    def get_observation_data(self) -> dict:
        shape_type, shape_params = self.shape.get_observation_params()
        dynamic_flag = 1.0 if self.type == ObstacleType.DYNAMIC else 0.0
        vel_x = self.velocity * self.direction[0] if self.type == ObstacleType.DYNAMIC else 0.0
        vel_y = self.velocity * self.direction[1] if self.type == ObstacleType.DYNAMIC else 0.0
        return {
            'x': self.x,
            'y': self.y,
            'shape_type': shape_type,
            'shape_params': shape_params,
            'dynamic_flag': dynamic_flag,
            'vel_x': vel_x,
            'vel_y': vel_y,
            'bounding_box': self.bounding_box
        }

    def update(self, dt: float = 1.0, bounds: tuple = (0, 0, 500, 500)):
        """
        Updates position and handles boundary bouncing using either its specific
        bounding_box or the global bounds.
        """
        if self.type != ObstacleType.DYNAMIC or self.velocity < 1e-6:
            return

        # Determine which bounds to use
        if self.bounding_box is not None:
            active_bounds = self.bounding_box
            # print(f"Using obstacle bounds: {active_bounds}") # Debug print
        else:
            active_bounds = bounds # Fallback to global bounds
            # print(f"Using global bounds: {active_bounds}") # Debug print

        min_x, min_y, max_x, max_y = active_bounds

        # Ensure bounds are logical
        if min_x >= max_x or min_y >= max_y:
             print(f"Warning: Invalid bounds for obstacle: {active_bounds}. Skipping update.")
             return

        new_x = self.x + self.velocity * self.direction[0] * dt
        new_y = self.y + self.velocity * self.direction[1] * dt

        # --- Boundary Bouncing (using active_bounds) ---
        bounced = False
        # Estimate boundary based on centroid for simplicity
        est_radius = self.shape.get_effective_radius()

        # Check X boundaries
        if new_x - est_radius < min_x:
            new_x = min_x + est_radius # Correct position
            if self.direction[0] < 0: # Only reverse if moving towards boundary
                self.direction[0] *= -1
                bounced = True
        elif new_x + est_radius > max_x:
            new_x = max_x - est_radius # Correct position
            if self.direction[0] > 0: # Only reverse if moving towards boundary
                self.direction[0] *= -1
                bounced = True

        # Check Y boundaries
        if new_y - est_radius < min_y:
            new_y = min_y + est_radius # Correct position
            if self.direction[1] < 0: # Only reverse if moving towards boundary
                self.direction[1] *= -1
                bounced = True
        elif new_y + est_radius > max_y:
            new_y = max_y - est_radius # Correct position
            if self.direction[1] > 0: # Only reverse if moving towards boundary
                self.direction[1] *= -1
                bounced = True


        # Normalize direction again if bounced to prevent weirdness after multiple bounces
        # Only normalize if it's non-zero
        if bounced:
             norm = np.linalg.norm(self.direction)
             if norm > 1e-6:
                  self.direction /= norm
             # else: # Should not happen if velocity > 0 and direction wasn't zero initially
             #      self.direction = np.array([1.0, 0.0]) # Reset to arbitrary direction? Or keep zero?

        # Clamp position strictly within bounds (especially important for bounding_box)
        # This prevents drifting slightly outside due to radius approximation
        self.x = np.clip(new_x, min_x + est_radius, max_x - est_radius)
        self.y = np.clip(new_y, min_y + est_radius, max_y - est_radius)

        # Ensure x/y don't get stuck exactly at boundary if radius is large compared to box
        self.x = max(min_x + est_radius, self.x)
        self.x = min(max_x - est_radius, self.x)
        self.y = max(min_y + est_radius, self.y)
        self.y = min(max_y - est_radius, self.y)

        # If somehow velocity is non-zero but direction became zero (e.g., exactly hitting a corner perfectly?)
        # Give it a nudge to prevent getting stuck. This is unlikely but a safeguard.
        if self.velocity > 1e-6 and np.linalg.norm(self.direction) < 1e-6:
             # print("Warning: Dynamic obstacle direction became zero. Resetting direction.")
             self.direction = np.array([1.0, 0.0]) # Or sample a random direction


    def get_render_patch(self, alpha: float = 0.6, zorder: int = 3) -> patches.Patch:
        return super().get_render_patch(color='darkorange', alpha=alpha, zorder=zorder)

# --- END OF FILE obstacle.py ---