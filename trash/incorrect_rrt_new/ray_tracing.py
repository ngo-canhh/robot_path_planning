import math
from shape import Circle, Rectangle

class RayTracing:
    """
    Class for performing ray tracing collision checks with obstacles.
    """

    def __init__(self, obstacles):
        """
        Initialize RayTracing with a list of obstacles.

        Args:
            obstacles (list of Obstacle): List of static obstacle objects to check for collision.
        """
        self.obstacles = obstacles # Renamed to obstacles, RayTracing doesn't need to know if they are static

    def check_ray_collision(self, start_point, end_point):
        """
        Check if a ray from start_point to end_point collides with any obstacle.
        Supports Circle and Rectangle shapes.

        Args:
            start_point (tuple): Start coordinates of the ray (x, y).
            end_point (tuple): End coordinates of the ray (x, y).

        Returns:
            bool: True if collision detected, False otherwise.
        """
        for obstacle in self.obstacles:
            if isinstance(obstacle.shape, Circle):
                if self._check_line_circle_intersection(start_point, end_point, obstacle):
                    return True
            elif isinstance(obstacle.shape, Rectangle):
                if self._ray_rectangle_intersection(start_point, end_point, obstacle):
                    return True
            elif hasattr(obstacle.shape, 'vertices'): # Check for vertices for polygon-like shapes
                if self._ray_polygon_intersection(start_point, end_point, obstacle):
                    return True
            # Bezier surface (temporarily ignored, can be added if needed)
        return False

    def _ray_round_face_intersection(self, start_point, end_point, obstacle):
        """
        Check intersection between ray and a round face (Circle shape).
        Based on equations (6), (7), (8), (9) in the paper.

        Args:
            start_point (tuple): Ray start point (x, y).
            end_point (tuple): Ray end point (x, y).
            obstacle (StaticObstacle): Static obstacle with a Circle shape.

        Returns:
            bool: True if ray intersects the circle, False otherwise.
        """
        center = (obstacle.shape.x, obstacle.shape.y)  # Circle center O(x0, y0, z0) - in 2D just (x0, y0)
        radius = obstacle.shape.radius # Radius r
        normal_vector = (0, 0, 1) # Normal vector of the plane containing the circle (simplified in 2D)
        origin = (0,0,0) # Origin of the world coordinate system, can be (0,0) in 2D

        Ro = start_point + (0,) # Robot position (viewpoint) Ro (r1, r2, r3) - add z=0 for 2D
        D = (end_point[0] - start_point[0], end_point[1] - start_point[1], 0) # Ray direction vector D (d1, d2, d3) - add z=0 for 2D
        D_magnitude = math.sqrt(D[0]**2 + D[1]**2 + D[2]**2)
        if D_magnitude == 0:
            return False # Zero length ray, no collision

        D_unit = (D[0]/D_magnitude, D[1]/D_magnitude, D[2]/D_magnitude) # Normalize D

        O_3d = center + (0,) # Circle center O (x0, y0, z0) - add z=0 for 2D
        N = normal_vector # Normal vector N (x1, y1, z1) - simplified in 2D

        # Equation (8): (D.N)t + (Ro - O).N = 0
        D_dot_N = sum(D_unit[i] * N[i] for i in range(3))
        Ro_minus_O_dot_N = sum((Ro[i] - O_3d[i]) * N[i] for i in range(3))

        if D_dot_N == 0:
            return False # Ray parallel to plane, no intersection (or on the plane)

        t = -Ro_minus_O_dot_N / D_dot_N # Equation (9) - intersection time to

        if t < 0: # Intersection point behind ray start, ignore
            return False

        # Calculate intersection point P = D*t + Ro
        P = (Ro[0] + D_unit[0] * t, Ro[1] + D_unit[1] * t) # Need only x, y in 2D

        # Check if P is inside the circle - Equation (7): (P-O).(P-O) <= r^2
        distance_sq_PO = (P[0] - center[0])**2 + (P[1] - center[0])**2 # Corrected center index
        if distance_sq_PO <= radius**2:
            return True # Intersection point inside circle
        return False # Intersection point outside circle or no valid intersection

    import math

    def _check_line_circle_intersection(self, start_point, end_point, obstacle):
        """
        Kiểm tra xem đoạn thẳng có giao điểm với hình tròn hay không.

        Args:
            start_point (tuple): Tọa độ điểm đầu của đoạn thẳng (x1, y1).
            end_point (tuple): Tọa độ điểm cuối của đoạn thẳng (x2, y2).
            obstacle (StaticObstacle): Đối tượng chướng ngại vật có hình dạng là Circle.

        Returns:
            bool: True nếu đoạn thẳng giao với hình tròn, False nếu không.
        """
        # Trích xuất center và radius từ obstacle.shape
        center = (obstacle.shape.x, obstacle.shape.y)
        radius = obstacle.shape.radius

        # Hàm tính khoảng cách từ một điểm đến tâm hình tròn
        def distance_to_center(point):
            return math.hypot(point[0] - center[0], point[1] - center[1])

        # Bước 1: Kiểm tra nếu start_point hoặc end_point nằm trong hình tròn
        if distance_to_center(start_point) <= radius or distance_to_center(end_point) <= radius:
            return True

        # Vector từ start_point đến end_point
        d = (end_point[0] - start_point[0], end_point[1] - start_point[1])
        # Vector từ start_point đến tâm hình tròn
        c = (center[0] - start_point[0], center[1] - start_point[1])

        # Tính tích vô hướng để tìm hệ số chiếu
        d_dot_d = d[0]**2 + d[1]**2
        if d_dot_d == 0:  # Trường hợp start_point và end_point trùng nhau
            return False  # Đã kiểm tra ở trên, cả hai điểm nằm ngoài

        proj = (c[0] * d[0] + c[1] * d[1]) / d_dot_d
        # Giới hạn t trong khoảng [0, 1] để điểm chiếu nằm trên đoạn thẳng
        t = max(0, min(1, proj))

        # Tính tọa độ điểm chiếu
        p = (start_point[0] + t * d[0], start_point[1] + t * d[1])

        # Tính khoảng cách từ tâm đến điểm chiếu
        dist = math.hypot(p[0] - center[0], p[1] - center[1])

        # Nếu khoảng cách nhỏ hơn hoặc bằng bán kính, có giao điểm
        return dist <= radius

    def _ray_rectangle_intersection(self, start_point, end_point, obstacle):
        """
        Check intersection between ray and rectangle (Rectangle shape).

        Args:
            start_point (tuple): Ray start point (x, y).
            end_point (tuple): Ray end point (x, y).
            obstacle (StaticObstacle): Static obstacle with a Rectangle shape.

        Returns:
            bool: True if ray intersects the rectangle, False otherwise.
        """
        # Lấy thông tin hình chữ nhật từ obstacle.shape
        rect = obstacle.shape
        rect_min_x = rect.x
        rect_min_y = rect.y
        rect_max_x = rect.x + rect.width
        rect_max_y = rect.y + rect.height

        # Hàm kiểm tra xem một điểm có nằm trong hình chữ nhật hay không
        def is_inside(x, y):
            return rect_min_x <= x <= rect_max_x and rect_min_y <= y <= rect_max_y

        # Kiểm tra nếu start_point hoặc end_point nằm trong hình chữ nhật
        if is_inside(*start_point) or is_inside(*end_point):
            return True

        # Định nghĩa bốn cạnh của hình chữ nhật dưới dạng các cặp điểm
        edges = [
            ((rect_min_x, rect_min_y), (rect_min_x, rect_max_y)),  # Cạnh trái
            ((rect_max_x, rect_min_y), (rect_max_x, rect_max_y)),  # Cạnh phải
            ((rect_min_x, rect_min_y), (rect_max_x, rect_min_y)),  # Cạnh dưới
            ((rect_min_x, rect_max_y), (rect_max_x, rect_max_y))   # Cạnh trên
        ]

        # Hàm kiểm tra giao điểm giữa hai đoạn thẳng
        def segments_intersect(seg1_start, seg1_end, seg2_start, seg2_end):
            def ccw(A, B, C):
                # Tính tích có hướng để xác định vị trí tương đối
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

            A, B = seg1_start, seg1_end
            C, D = seg2_start, seg2_end
            # Kiểm tra xem hai đoạn thẳng có giao nhau hay không
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        # Kiểm tra giao điểm của đoạn thẳng với từng cạnh của hình chữ nhật
        for edge_start, edge_end in edges:
            if segments_intersect(start_point, end_point, edge_start, edge_end):
                return True

        # Nếu không có giao điểm nào được tìm thấy, trả về False
        return False

    def _ray_polygon_intersection(self, start_point, end_point, obstacle):
        """
        Check intersection between ray and polygon (Polygon shape).

        Args:
            start_point (tuple): Ray start point (x, y).
            end_point (tuple): Ray end point (x, y).
            obstacle (StaticObstacle): Static obstacle with a Polygon shape (assumed to have 'vertices' attribute).

        Returns:
            bool: True if ray intersects the polygon, False otherwise.
        """
        polygon_vertices = obstacle.shape.vertices  # Assuming polygon has a 'vertices' attribute

        # Calculate normal vector of the polygon's plane
        if len(polygon_vertices) < 3:
            return False  # Not a valid polygon

        v1, v2, v3 = polygon_vertices[0], polygon_vertices[1], polygon_vertices[2]
        plane_normal = self._calculate_normal_vector(v1, v2, v3)
        if plane_normal is None:
            return False  # Degenerate polygon

        # Find intersection point between ray and plane
        intersection = self._ray_plane_intersection(start_point, end_point, plane_normal, v1)
        if intersection is None:
            return False  # No intersection with plane

        # Check if intersection point is inside the polygon
        return self._is_point_in_polygon_2d(intersection[:2], polygon_vertices) # Consider only 2D coordinates

    def _calculate_normal_vector(self, v1, v2, v3):
        """
        Calculate normal vector of plane defined by 3 points v1, v2, v3 (3D).
        In 2D, can return (0, 0, 1) or (0, 0, -1) depending on orientation.

        Args:
            v1 (tuple): Point 1 (x, y, z).
            v2 (tuple): Point 2 (x, y, z).
            v3 (tuple): Point 3 (x, y, z).

        Returns:
            tuple: Normal vector (nx, ny, nz) or None if degenerate.
        """
        v1xv2 = ( (v2[0] - v1[0]), (v2[1] - v1[1]), (v2[2] - v1[2]) )
        v1xv3 = ( (v3[0] - v1[0]), (v3[1] - v1[1]), (v3[2] - v1[2]) )

        normal_x = v1xv2[1] * v1xv3[2] - v1xv2[2] * v1xv3[1]
        normal_y = v1xv2[2] * v1xv3[0] - v1xv2[0] * v1xv3[2]
        normal_z = v1xv2[0] * v1xv3[1] - v1xv2[1] * v1xv3[0]

        normal_magnitude = math.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        if normal_magnitude == 0:
            return None # Degenerate polygon, no normal vector

        return (normal_x/normal_magnitude, normal_y/normal_magnitude, normal_z/normal_magnitude) # Normalize

    def _is_point_in_polygon_2d(self, point, polygon_vertices):
        """
        Check if a 2D point is inside a 2D polygon using ray casting algorithm.
        (Placeholder - Implement ray casting or winding number algorithm for accuracy)

        Args:
            point (tuple): Point to check (x, y).
            polygon_vertices (list of tuples): List of polygon vertices [(x1, y1), (x2, y2), ...].

        Returns:
            bool: True if point is inside polygon, False otherwise.
        """
        # Placeholder - Implement point-in-polygon algorithm (e.g., ray casting)
        # This is just a simple bounding box check for now, replace with accurate algorithm
        min_x = min(v[0] for v in polygon_vertices)
        max_x = max(v[0] for v in polygon_vertices)
        min_y = min(v[1] for v in polygon_vertices)
        max_y = max(v[1] for v in polygon_vertices)
        x_point, y_point = point
        return (x_point > min_x and x_point < max_x and y_point > min_y and y_point < max_y) # Bounding box check (VERY simple)

    def _ray_plane_intersection(self, ray_start, ray_end, plane_normal, plane_point):
        """
        Find intersection point between ray and plane.

        Args:
            ray_start (tuple): Ray start point (x, y, z).
            ray_end (tuple): Ray end point (x, y, z).
            plane_normal (tuple): Plane normal vector (nx, ny, nz).
            plane_point (tuple): A point on the plane (px, py, pz).

        Returns:
            tuple: Intersection point (x, y, z) or None if no intersection.
        """
        # Ray direction
        ray_dir = (ray_end[0] - ray_start[0], ray_end[1] - ray_start[1], ray_end[2] - ray_start[2])

        # Dot product of ray direction and plane normal
        denom = sum(ray_dir[i] * plane_normal[i] for i in range(3))
        if abs(denom) < 1e-6:  # Ray is parallel to plane
            return None

        # Vector from ray start to plane point
        ray_to_plane = (plane_point[0] - ray_start[0], plane_point[1] - ray_start[1], plane_point[2] - ray_start[2])

        # Calculate t (parameter along the ray)
        t = sum(ray_to_plane[i] * plane_normal[i] for i in range(3)) / denom

        if t < 0:  # Intersection is behind the ray start
            return None

        # Calculate intersection point
        intersection = (
            ray_start[0] + t * ray_dir[0],
            ray_start[1] + t * ray_dir[1],
            ray_start[2] + t * ray_dir[2]
        )
        return intersection

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from obstacle import StaticObstacle
    ray_tracer = RayTracing([StaticObstacle(Circle(25, 25, 15))])
    print(ray_tracer.check_ray_collision((0, 0), (50, 50)))  # Should return True