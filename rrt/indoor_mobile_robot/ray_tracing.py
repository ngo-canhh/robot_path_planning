import math
from shape import Circle, Rectangle
class RayTracing:
    
    def __init__(self, obstacles):
        """Initialize RayTracing with list of static obstacles."""
        self.static_obstacles = obstacles

    def check_ray_collision(self, start_point, end_point):
        """
        Check if ray from start_point to end_point collides with any static obstacle.
        Uses ray projection and plane equation method.
        """
        for obstacle in self.static_obstacles:
            if isinstance(obstacle.shape, Circle):
                if self._ray_round_face_intersection(start_point, end_point, obstacle):
                    return True
            elif isinstance(obstacle.shape, Rectangle):
                if self._ray_rectangle_intersection(start_point, end_point, obstacle):
                    return True
            elif obstacle.shape.shape == 'polygon': # Assuming polygon shape has 'polygon' attribute
                if self._ray_polygon_intersection(start_point, end_point, obstacle):
                    return True
            # Bezier surface (temporarily ignored, can be added if needed)
        return False

    def _ray_round_face_intersection(self, start_point, end_point, obstacle):
        """
        Check intersection between ray and round face (Circle shape).
        Based on equations (6), (7), (8), (9) in the paper.
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
        distance_sq_PO = (P[0] - center[0])**2 + (P[1] - center[1])**2
        if distance_sq_PO <= radius**2:
            return True # Intersection point inside circle
        return False # Intersection point outside circle or no valid intersection

    def _ray_rectangle_intersection(self, start_point, end_point, obstacle):
        """
        Check intersection between ray and rectangle (Rectangle shape).
        Implementation needed - similar to circle but with rectangle boundaries.
        """
        # Placeholder - Implement ray-rectangle intersection logic if needed
        return False  # Placeholder, implement proper logic

    def _ray_polygon_intersection(self, start_point, end_point, obstacle):
        """
        Check intersection between ray and polygon (Polygon shape).
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
        return self._is_point_in_polygon(intersection, polygon_vertices)

    def _calculate_normal_vector(self, v1, v2, v3):
        """
        Calculate normal vector of plane defined by 3 points v1, v2, v3 (3D).
        In 2D, can return (0, 0, 1) or (0, 0, -1) depending on orientation.
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
        Check if a 2D point is inside a 2D polygon (e.g., ray casting algorithm).
        (Placeholder - Implement ray casting or winding number algorithm)
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
    
