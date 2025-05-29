import math
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import matplotlib.patches as patches
from components.shape import Shape, Circle, Rectangle

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
                 self.direction = np.array([0.0, 0.0])
        else:
            self.direction = np.array([0.0, 0.0])

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

    def get_observation_data(self) -> list:
        """
        Returns obstacle data formatted for the environment's observation space.
        Format: [x, y, shape_type, p1, p2, p3, is_dynamic, vel_x, vel_y]
        """
        shape_type, p1, p2, p3 = self.shape.get_observation_params()
        dynamic_flag = 1.0 if self.type == ObstacleType.DYNAMIC else 0.0
        vel_x = self.velocity * self.direction[0] if self.type == ObstacleType.DYNAMIC else 0.0
        vel_y = self.velocity * self.direction[1] if self.type == ObstacleType.DYNAMIC else 0.0
        return [self.x, self.y, shape_type, p1, p2, p3, dynamic_flag, vel_x, vel_y]

    def reset_to_initial(self):
         """Resets obstacle to its starting position (useful for planner map)."""
         self.x = self.initial_x
         self.y = self.initial_y
         # Should velocity/direction also be reset if they could change? Assume not for now.


class StaticObstacle(Obstacle):
    """Obstacle that does not move."""
    def __init__(self, x: float, y: float, shape: Shape):
        super().__init__(x, y, shape, ObstacleType.STATIC)

    def update(self, dt: float = 1.0, bounds: tuple = (0, 0, 500, 500)):
        # Static obstacles don't move
        pass

    def get_render_patch(self, alpha: float = 0.6, zorder: int = 3) -> patches.Patch:
        return super().get_render_patch(color='dimgray', alpha=alpha, zorder=zorder)


class DynamicObstacle(Obstacle):
    """Obstacle that moves with a constant velocity, bouncing off boundaries."""
    def __init__(self, x: float, y: float, shape: Shape, velocity: float, direction: np.ndarray):
        super().__init__(x, y, shape, ObstacleType.DYNAMIC, velocity, direction)

    def update(self, dt: float = 1.0, bounds: tuple = (0, 0, 500, 500)):
        """Updates position and handles boundary bouncing."""
        if self.type != ObstacleType.DYNAMIC or self.velocity < 1e-6:
            return

        min_x, min_y, max_x, max_y = bounds

        new_x = self.x + self.velocity * self.direction[0] * dt
        new_y = self.y + self.velocity * self.direction[1] * dt

        # --- Boundary Bouncing ---
        # This is approximate for non-circular shapes. We use the center point.
        # A more accurate check would involve the shape's bounding box or corners.
        # Using center point for simplicity here.
        bounced = False
        # Estimate boundary based on centroid for simplicity
        # TODO: Improve boundary check for non-circular shapes (use bounding box?)
        est_radius = 0 # Placeholder - should use shape bounds
        if isinstance(self.shape, Circle):
            est_radius = self.shape.radius
        elif isinstance(self.shape, Rectangle):
            # Use half-diagonal as a rough bounding radius
            est_radius = 0.5 * math.sqrt(self.shape.width**2 + self.shape.height**2)

        if new_x - est_radius < min_x:
            new_x = min_x + est_radius
            self.direction[0] *= -1
            bounced = True
        elif new_x + est_radius > max_x:
            new_x = max_x - est_radius
            self.direction[0] *= -1
            bounced = True

        if new_y - est_radius < min_y:
            new_y = min_y + est_radius
            self.direction[1] *= -1
            bounced = True
        elif new_y + est_radius > max_y:
            new_y = max_y - est_radius
            self.direction[1] *= -1
            bounced = True

        # Normalize direction again if bounced to prevent drift from multiple bounces
        if bounced:
             norm = np.linalg.norm(self.direction)
             if norm > 1e-6:
                  self.direction /= norm
             else: # Should not happen if velocity > 0
                  self.direction = np.array([1.0, 0.0]) # Reset to arbitrary direction

        self.x = new_x
        self.y = new_y


    def get_render_patch(self, alpha: float = 0.6, zorder: int = 3) -> patches.Patch:
        return super().get_render_patch(color='darkorange', alpha=alpha, zorder=zorder)
