import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import random

from components.shape import Circle, Rectangle
from enum import Enum

class ObstacleType(Enum):
    STATIC = 0
    DYNAMIC = 1

def find_image_files(image_dir):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = []
    try:
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_files.append(os.path.join(root, file))
    except Exception as e:
        print(f"Error walking directory {image_dir}: {e}")
    return image_files

class Obstacle:
    def __init__(self, x, y, shape, image_path=None):
        self.x = x
        self.y = y
        self.shape = shape
        self.image_path = image_path
        self.type = None

    def check_collision(self, x, y, radius):
        return self.shape.check_collision(x, y, radius, self.x, self.y)

    def get_observation_data(self):
        raise NotImplementedError("Subclasses must implement get_observation_data")

    def get_render_patch(self, ax, alpha=0.6, zorder=3):
        if self.image_path and os.path.exists(self.image_path):
            try:
                img = plt.imread(self.image_path)
                if isinstance(self.shape, Circle):
                    imagebox = OffsetImage(img, zoom=min(self.shape.radius * 2 / max(img.shape), 1.0), alpha=alpha)
                    ab = AnnotationBbox(imagebox, (self.x, self.y), frameon=False, zorder=zorder)
                    return ab
                elif isinstance(self.shape, Rectangle):
                    imagebox = OffsetImage(img, zoom=min(self.shape.width / max(img.shape), self.shape.height / max(img.shape)), alpha=alpha)
                    ab = AnnotationBbox(imagebox, (self.x, self.y), frameon=False, zorder=zorder)
                    return ab
            except Exception as e:
                print(f"Error loading image {self.image_path}: {e}")
        return self.shape.get_patch(self.x, self.y, ax, alpha=alpha, zorder=zorder)

    def update(self, dt, bounds):
        pass  # Base class does nothing

class StaticObstacle(Obstacle):
    def __init__(self, x, y, shape, image_path=None):
        super().__init__(x, y, shape, image_path)
        self.type = ObstacleType.STATIC

    def get_observation_data(self):
        if isinstance(self.shape, Circle):
            return [self.x, self.y, Circle.SHAPE_TYPE_ENUM, self.shape.radius, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif isinstance(self.shape, Rectangle):
            return [self.x, self.y, Rectangle.SHAPE_TYPE_ENUM, self.shape.width, self.shape.height, self.shape.angle, 0.0, 0.0, 0.0]
        else:
            raise ValueError(f"Unsupported shape type: {type(self.shape)}")

    def update(self, dt, bounds):
        # Static obstacles do not move
        pass

class DynamicObstacle(Obstacle):
    def __init__(self, x, y, shape, image_path=None, velocity=0.0, direction=np.array([0.0, 0.0])):
        super().__init__(x, y, shape, image_path)
        self.velocity = velocity
        self.direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 1e-6 else np.array([0.0, 0.0])
        self.type = ObstacleType.DYNAMIC

    def get_observation_data(self):
        vel_x = self.velocity * self.direction[0]
        vel_y = self.velocity * self.direction[1]
        if isinstance(self.shape, Circle):
            return [self.x, self.y, Circle.SHAPE_TYPE_ENUM, self.shape.radius, 0.0, 0.0, 1.0, vel_x, vel_y]
        elif isinstance(self.shape, Rectangle):
            return [self.x, self.y, Rectangle.SHAPE_TYPE_ENUM, self.shape.width, self.shape.height, self.shape.angle, 1.0, vel_x, vel_y]
        else:
            raise ValueError(f"Unsupported shape type: {type(self.shape)}")

    def update(self, dt, bounds):
        x_min, y_min, x_max, y_max = bounds
        dx = self.velocity * self.direction[0] * dt
        dy = self.velocity * self.direction[1] * dt
        new_x = self.x + dx
        new_y = self.y + dy

        placement_radius = 0
        if isinstance(self.shape, Circle):
            placement_radius = self.shape.radius
        elif isinstance(self.shape, Rectangle):
            placement_radius = 0.5 * math.sqrt(self.shape.width**2 + self.shape.height**2)

        if new_x - placement_radius < x_min or new_x + placement_radius > x_max:
            self.direction[0] = -self.direction[0]
            new_x = self.x
        if new_y - placement_radius < y_min or new_y + placement_radius > y_max:
            self.direction[1] = -self.direction[1]
            new_y = self.y

        self.x = new_x
        self.y = new_y