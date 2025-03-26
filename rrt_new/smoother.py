from abc import ABC, abstractmethod
from ray_tracing import RayTracing
from waiting_rule import WaitingRule
from obstacle import StaticObstacle, Rectangle
import numpy as np


class Smoother(ABC):
    @abstractmethod
    def smooth_path(self, path):
        pass
    
class ShortcutSmoother(Smoother):
     def __init__(self, ray_tracer):
         self.ray_tracer = ray_tracer
        
     def smooth_path(self, path):
        """
        Smooths the given path using the shortcut smoothing algorithm.

        Args:
            path: The original path as a list of points (tuples of floats).
            ray_tracer: An instance of the RayTracing class for collision checking.

        Returns:
            The smoothed path as a list of points.
        """
        smoothed_path = list(path) # copy path (đảm bảo là list để có thể sửa đổi)
        print(f"Start smoothing {smoothed_path}")
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(smoothed_path) - 2:
                point1 = smoothed_path[i]
                point2 = smoothed_path[i + 2]
                # segment = create_segment(point1, point2)
                if not self.ray_tracer.check_ray_collision(point1, point2): # Sử dụng hàm kiểm tra va chạm cho lưới
                    a = smoothed_path.pop(i+1)
                    print(a)
                    changed = True
                else:
                    i += 1
        return smoothed_path
     
class BezierSmooth(Smoother):
    # Lưu ý: Các phương thức bên dưới cần được chuyển thành phương thức instance (thêm self) hoặc dùng @staticmethod nếu không dùng biến instance.
    def bezier_curve(self, P0, P1, P2, P3, num_points=50):
        t = np.linspace(0, 1, num_points)
        curve = np.outer((1-t)**3, P0) + np.outer(3*(1-t)**2*t, P1) \
            + np.outer(3*(1-t)*t**2, P2) + np.outer(t**3, P3)
        return curve

    def smooth_path_bezier(self, raw_path, num_points_per_segment=50):
        raw_path = np.array(raw_path)
        smooth_path = []

        if len(raw_path) < 4:
            return raw_path

        for i in range(0, len(raw_path) - 3, 3):
            P0 = raw_path[i]
            P1 = raw_path[i+1]
            P2 = raw_path[i+2]
            P3 = raw_path[i+3]
            segment = self.bezier_curve(P0, P1, P2, P3, num_points=num_points_per_segment)
            if i != 0:
                segment = segment[1:]
            smooth_path.extend(segment.tolist())
        return np.array(smooth_path)

    def smooth_path(self, path):
        """
        Cho phương thức smooth_path trong BezierSmooth,
        bạn có thể gọi hàm smooth_path_bezier ở đây.
        """
        return self.smooth_path_bezier(path)

class Pipeline(Smoother):
    def __init__(self, smoothers):
        self.smoothers = smoothers
    
    def smooth_path(self, path):
        new_path = list(path)
        for smoother in self.smoothers:
            new_path = smoother.smooth_path(new_path)
        return new_path
        