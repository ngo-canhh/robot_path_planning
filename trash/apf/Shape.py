from abc import ABC, abstractmethod
import numpy as np

class Shape(ABC):
  def __init__(self, x, y):
    self.x = x
    self.y = y

  @abstractmethod
  def get_centroid(self):
    pass

class Circle(Shape):
  def __init__(self, x, y, radius):
    super().__init__(x, y)
    self.radius = radius

  def draw(self):
    pass

  def get_min_distance(self, x, y):
    return np.hypot(self.x - x, self.y - y) - self.radius
  
  def get_centroid(self):
    return self.x, self.y
  
class Rectangle(Shape):
  def __init__(self, x, y, width, height):
    super().__init__(x, y)
    self.width = width
    self.height = height

  def draw(self):
    pass

  def get_min_distance(self, x, y):
    dx = max(self.x - x, x - self.x - self.width, 0)
    dy = max(self.y - y, y - self.y - self.height, 0)
    return np.hypot(dx, dy)
  
  def get_centroid(self):
    return self.x + self.width / 2, self.y + self.height / 2