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
                    # print(a)
                    changed = True
                else:
                    i += 1
        return smoothed_path
     
class BezierSmooth(Smoother):
    def __init__(self, ray_tracer=None):
        self.ray_tracer = ray_tracer
        
    # Lưu ý: Các phương thức bên dưới cần được chuyển thành phương thức instance (thêm self) hoặc dùng @staticmethod nếu không dùng biến instance.
    def bezier_curve(self, P0, P1, P2, P3, num_points=50):
        t = np.linspace(0, 1, num_points)
        curve = np.outer((1-t)**3, P0) + np.outer(3*(1-t)**2*t, P1) \
            + np.outer(3*(1-t)*t**2, P2) + np.outer(t**3, P3)
        return curve

    def smooth_path_bezier(self, raw_path, num_points_per_segment=50):
        raw_path = np.array(raw_path + [raw_path[-1]])
        smooth_path = []

        print(f'len: {len(raw_path)}')
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
        smooth_path.append(raw_path[-1])
        return np.array(smooth_path)
    
    import numpy as np

    def check_bezier_collision(self, P0, P1, P2, P3, num_points=50):
        t = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 2))
        for i in range(num_points):
            t_val = t[i]
            curve[i] = (1-t_val)**3 * P0 + 3*(1-t_val)**2*t_val * P1 + \
                    3*(1-t_val)*t_val**2 * P2 + t_val**3 * P3
        for i in range(num_points - 1):
            if self.ray_tracer.check_ray_collision(curve[i], curve[i+1]):
                return True  # Có va chạm
        return False  # Không va chạm

    # Sử dụng trong smooth_path_bezier
    def strong_smooth_path_bezier(self, raw_path, num_points_per_segment=20):
        smooth_path = []
        raw_path = np.array(raw_path)
        for i in range(0, len(raw_path) - 3, 3):
            P0, P1, P2, P3 = raw_path[i], raw_path[i+1], raw_path[i+2], raw_path[i+3]
            if not self.check_bezier_collision(P0, P1, P2, P3):
                t = np.linspace(0, 1, num_points_per_segment)
                for t_val in t:
                    point = (1-t_val)**3 * P0 + 3*(1-t_val)**2*t_val * P1 + \
                            3*(1-t_val)*t_val**2 * P2 + t_val**3 * P3
                    smooth_path.append(point)
            else:
                # Nếu có va chạm, giữ nguyên các điểm ban đầu
                smooth_path.extend([P0, P1, P2, P3])
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


if __name__ == '__main__':
    # Test the ShortcutSmoother
    ray_tracer = RayTracing([StaticObstacle(Rectangle(10, 10, 40, 5))])
    smoother = ShortcutSmoother(ray_tracer)
    path = [(0, 0), (10, 20), (20, 20), (30, 30), (40, 40), (50, 50)]
    smoothed_path = smoother.smooth_path(path)
    print(f"Smoothed path: {smoothed_path}")

    # Test the BezierSmooth
    smoother = BezierSmooth(ray_tracer)
    smoothed_path = smoother.smooth_path(path)
    print(f"Smoothed path: {smoothed_path}")


    # Vizualize the smoothed path
    import matplotlib.pyplot as plt
    # Plot the obstacle
    obstacle = ray_tracer.obstacles[0]
    rectangle = obstacle.shape
    x1, y1 = rectangle.x, rectangle.y
    x2, y2 = x1 + rectangle.width, y1 + rectangle.height
    plt.fill([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'gray', alpha=0.5, label="Obstacle")
    plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r', label="Original path")
    plt.plot([x for (x, y) in smoothed_path], [y for (x, y) in smoothed_path], '-b', label="Smoothed path")
    plt.grid(True)
    plt.legend()
    plt.show()