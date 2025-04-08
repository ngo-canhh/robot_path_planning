import numpy as np

# --- RRTPathPlanner ---

class RRTPathPlanner:
    def __init__(self, env_width, env_height, robot_radius):
        self.width = env_width
        self.height = env_height
        self.robot_radius = robot_radius
        self.step_size = 30
        self.goal_sample_rate = 0.15
        self.max_iterations = 3000
        self.min_dist_to_goal = self.robot_radius * 2

        self.nodes = []
        self.parents = []
        self.planning_obstacles = [] # List of Obstacle objects for planning

    def plan_path(self, start_x, start_y, goal_x, goal_y, obstacles_for_planning: list):
        """ Plan path using RRT with Obstacle objects. """
        # Use the provided list of obstacles (assumed static or reset to initial)
        self.planning_obstacles = obstacles_for_planning

        self.nodes = [(start_x, start_y)]
        self.parents = [-1]
        path_found = False
        goal_node_idx = -1

        for i in range(self.max_iterations):
            # 1. Sample point
            if np.random.random() < self.goal_sample_rate:
                rnd_point = (goal_x, goal_y)
            else:
                rnd_point = (
                    np.random.uniform(self.robot_radius, self.width - self.robot_radius),
                    np.random.uniform(self.robot_radius, self.height - self.robot_radius)
                )

            # 2. Find nearest node
            nearest_idx = self._find_nearest(self.nodes, rnd_point)
            nearest_node = self.nodes[nearest_idx]

            # 3. Steer
            new_node = self._steer(nearest_node, rnd_point, self.step_size)

            # Check bounds
            if not (self.robot_radius <= new_node[0] <= self.width - self.robot_radius and
                    self.robot_radius <= new_node[1] <= self.height - self.robot_radius):
                 continue

            # 4. Check collision for the new segment using Obstacle.intersects_segment
            if self._is_segment_collision_free(nearest_node, new_node):
                # 5. Add node and edge
                self.nodes.append(new_node)
                self.parents.append(nearest_idx)
                new_node_idx = len(self.nodes) - 1

                # 6. Check goal reached
                dist_to_goal = np.linalg.norm(np.array(new_node) - np.array((goal_x, goal_y)))
                if dist_to_goal <= self.min_dist_to_goal:
                     # Try connecting the new node directly to the goal
                     if self._is_segment_collision_free(new_node, (goal_x, goal_y)):
                          self.nodes.append((goal_x, goal_y))
                          self.parents.append(new_node_idx)
                          goal_node_idx = len(self.nodes) - 1
                          print(f"RRT: Path found connecting near goal node in {i+1} iterations.")
                          path_found = True
                          break

        # Path Extraction
        path = []
        final_nodes = list(self.nodes)
        final_parents = list(self.parents)

        if path_found:
            current_idx = goal_node_idx
            while current_idx != -1:
                path.insert(0, self.nodes[current_idx])
                current_idx = self.parents[current_idx]
            # Optional Smoothing (using _is_segment_collision_free)
            # path = self._smooth_path(path) # Be cautious with smoothing complex shapes
        else:
            print(f"RRT: Failed to find path after {self.max_iterations} iterations.")

        return path, final_nodes, final_parents

    def _find_nearest(self, nodes, point):
        nodes_arr = np.array(nodes)
        point_arr = np.array(point)
        distances_sq = np.sum((nodes_arr - point_arr)**2, axis=1)
        return np.argmin(distances_sq)

    def _steer(self, from_node, to_node, step_size):
        from_arr = np.array(from_node)
        to_arr = np.array(to_node)
        vec = to_arr - from_arr
        dist = np.linalg.norm(vec)
        if dist <= step_size:
            return tuple(to_arr)
        else:
            unit_vec = vec / dist
            new_node_arr = from_arr + unit_vec * step_size
            return tuple(new_node_arr)

    def _is_segment_collision_free(self, from_node, to_node):
        """ Checks segment collision against planning_obstacles using intersects_segment. """
        # Check boundaries first (simple check on nodes)
        for node in [from_node, to_node]:
             x, y = node
             if not (self.robot_radius <= x <= self.width - self.robot_radius and
                      self.robot_radius <= y <= self.height - self.robot_radius):
                  return False # Node itself is out of bounds

        # Check line segment against obstacles
        for obstacle in self.planning_obstacles:
            # Use the obstacle's intersects_segment method, which delegates to the shape
            if obstacle.intersects_segment(from_node, to_node, self.robot_radius):
                 # print(f"RRT collision: Segment {from_node}->{to_node} intersects obs at ({obstacle.x:.1f},{obstacle.y:.1f})")
                 return False # Collision detected
        return True # Segment is collision-free

    def _smooth_path(self, path):
        """ Shortcut path using _is_segment_collision_free """
        if len(path) <= 2: return path
        smoothed_path = [path[0]]
        current_idx = 0
        while current_idx < len(path) - 1:
            best_next_idx = current_idx + 1
            for next_idx in range(len(path) - 1, current_idx + 1, -1):
                if self._is_segment_collision_free(path[current_idx], path[next_idx]):
                    best_next_idx = next_idx
                    break
            smoothed_path.append(path[best_next_idx])
            current_idx = best_next_idx
        return smoothed_path

