from abc import ABC, abstractmethod
from ray_tracing import RayTracing
from obstacle import StaticObstacle, Rectangle


class Smoother(ABC):
    @abstractmethod
    def smooth_path(self, path):
        pass
    
class ShortcutSmoother(Smoother):
        
     def smooth_path(self, path, ray_tracer: RayTracing):
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
                if not ray_tracer.check_ray_collision(point1, point2): # Sử dụng hàm kiểm tra va chạm cho lưới
                    a = smoothed_path.pop(i+1)
                    print(a)
                    changed = True
                else:
                    i += 1
        return smoothed_path
    
if __name__ == '__main__':
    # Test the ShortcutSmoother class
    path = [(5, 5), (55, 55)]
    ray_tracer = RayTracing([StaticObstacle(Rectangle(25, 10, 20, 30))])
    print(ray_tracer.check_ray_collision((5, 5), (55, 55)))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([x for x, y in path], [y for x, y in path], 'r-')
    rec = plt.Rectangle((25, 10), 20, 30, edgecolor='k', facecolor='k')
    plt.gca().add_patch(rec)
    plt.show()