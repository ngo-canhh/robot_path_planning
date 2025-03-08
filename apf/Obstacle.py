from abc import ABC, abstractmethod
from Shape import Shape, Circle, Rectangle

class Obstacle(ABC):
  @abstractmethod
  def draw(self):
    pass

  @abstractmethod
  def get_min_distance(self, x, y):
    pass

  @abstractmethod
  def get_centroid(self):
    pass


class StaticObstacle(Obstacle):
  def __init__(self, shape: Shape):
    self.shape = shape

  def draw(self):
    pass

  def get_min_distance(self, x, y):
    return self.shape.get_min_distance(x, y)
  
  def get_centroid(self):
    return self.shape.get_centroid()
  
class DynamicObstacle(Obstacle):
  def __init__(self, shape: Shape, vx, vy):
    self.shape = shape
    self.vx = vx
    self.vy = vy

  def draw(self):
    pass

  def get_min_distance(self, x, y):
    return self.shape.get_min_distance(x, y)
  
  def get_centroid(self):
    return self.shape.get_centroid()
  
  def move(self, dt):
    self.shape.x += self.vx * dt
    self.shape.y += self.vy * dt