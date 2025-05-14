import numpy as np
from typing import List, Tuple, Callable, Literal

# --- PathSmoother ---
class PathSmoother:
    """
    Encapsulates the path smoothing (shortcut) logic.
    Relies on an external function for collision checking.
    """
    def __init__(self):
        # Smoother could have its own parameters in the future if needed
        pass

    def smooth(self,
               path: List[Tuple[float, float]],
               collision_check_func: Callable[[Tuple[float, float], Tuple[float, float]], bool]
               ) -> List[Tuple[float, float]]:
        """
        Attempts to shortcut the path by connecting non-adjacent nodes
        if the direct segment is collision-free.

        Args:
            path: The initial path as a list of (x, y) tuples.
            collision_check_func: A function that takes two points (nodes)
                                   and returns True if the segment connecting
                                   them is collision-free, False otherwise.

        Returns:
            The smoothed path as a list of (x, y) tuples.
        """
        if len(path) <= 2:
            return path # Cannot smooth a path with 0, 1, or 2 points

        smoothed_path = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            best_next_idx = current_idx + 1 # Default to the very next node
            # Try to connect to nodes further down the path
            for next_idx in range(len(path) - 1, current_idx + 1, -1):
                # Check if the direct path from current_node to path[next_idx] is free
                if collision_check_func(path[current_idx], path[next_idx]):
                    # Found a valid shortcut
                    best_next_idx = next_idx
                    break # Take the furthest valid shortcut

            smoothed_path.append(path[best_next_idx])
            current_idx = best_next_idx # Jump to the node we connected to

        return smoothed_path


# --- RRTConnectPathPlanner ---

class RRTConnectPathPlanner:
    def __init__(self, env_width, env_height, robot_radius):
        self.width = env_width
        self.height = env_height
        self.robot_radius = robot_radius
        self.step_size = 30
        self.max_iterations = 3000
        self.min_dist_to_goal = self.robot_radius * 2
        self.goal = None

        # RRT-Connect uses two trees
        self._start_tree = {'nodes': [], 'parents': []}
        self._goal_tree = {'nodes': [], 'parents': []}
        self._planning_obstacles = []

        # Composition: Planner uses a smoother
        self._smoother = PathSmoother()
    
    def set_goal(self, goal_x: float, goal_y: float):
        # Ensure goal is within bounds
        if not (self.robot_radius <= goal_x <= self.width - self.robot_radius and
                self.robot_radius <= goal_y <= self.height - self.robot_radius):
            raise ValueError("Goal coordinates are out of bounds.")
        self.goal = (goal_x, goal_y)

    def plan_path(self,
                  start_x: float,
                  start_y: float,
                  goal_x: float,
                  goal_y: float,
                  obstacles_for_planning: list,
                  smooth_path: bool = False
                 ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[int]]:
        """
        Plan path using RRT-Connect, optionally smoothing the result.

        Args:
            start_x: Starting x-coordinate.
            start_y: Starting y-coordinate.
            goal_x: Goal x-coordinate.
            goal_y: Goal y-coordinate.
            obstacles_for_planning: A list of Obstacle objects to use for collision checking.
                                     These are assumed static for the duration of planning.
            smooth_path: If True, attempt to smooth the found path.

        Returns:
            A tuple containing:
            - path: List of (x, y) tuples representing the path (smoothed if requested).
                    Empty list if no path is found.
            - final_nodes: List of all nodes explored by both trees.
            - final_parents: List of parent indices for each node in final_nodes.
        """
        # --- Reset internal state for new planning ---
        self._planning_obstacles = obstacles_for_planning
        
        # Initialize trees
        self._start_tree = {'nodes': [(start_x, start_y)], 'parents': [-1]}
        self._goal_tree = {'nodes': [(goal_x, goal_y)], 'parents': [-1]}
        
        # Track which trees nodes belong to (for visualization and debugging)
        self._node_tree_map = {}  # Maps node index in combined tree to 'start' or 'goal'
        
        path_found = False
        connecting_node_a = None
        connecting_node_b = None
        connecting_idx_a = -1
        connecting_idx_b = -1
        
        for i in range(self.max_iterations):
            # Sample a random point
            if np.random.random() < 0.05:  # Lower bias than standard RRT
                rnd_point = (goal_x, goal_y)
            else:
                rnd_point = (
                    np.random.uniform(self.robot_radius, self.width - self.robot_radius),
                    np.random.uniform(self.robot_radius, self.height - self.robot_radius)
                )
                
            # Expand the trees (alternating between start and goal trees)
            if i % 2 == 0:
                tree_a, tree_b = self._start_tree, self._goal_tree
                tree_a_name, tree_b_name = 'start', 'goal'
            else:
                tree_a, tree_b = self._goal_tree, self._start_tree
                tree_a_name, tree_b_name = 'goal', 'start'
                
            # Grow tree_a towards random point
            new_node_idx = self._extend(tree_a, rnd_point)
            
            if new_node_idx != -1:  # If extension successful
                # Try to connect tree_b to the new node in tree_a
                new_node = tree_a['nodes'][new_node_idx]
                connect_result = self._connect(tree_b, new_node)
                
                if connect_result != -1:  # Connection successful
                    # Found a path connecting the trees
                    connecting_node_a = new_node
                    connecting_node_b = tree_b['nodes'][connect_result]
                    
                    # Store indices to avoid precision issues when looking them up later
                    if tree_a_name == 'start':
                        connecting_idx_a = new_node_idx
                        connecting_idx_b = connect_result
                    else:
                        connecting_idx_b = new_node_idx
                        connecting_idx_a = connect_result
                    
                    path_found = True
                    print(f"RRT-Connect: Path found in {i+1} iterations")
                    break
        
        # --- Path Extraction and Merging ---
        raw_path = []
        
        # Create combined tree for visualization (required for API compatibility)
        combined_nodes = list(self._start_tree['nodes']) + list(self._goal_tree['nodes'])
        
        # Adjust parent indices for the goal tree to account for merged node list
        start_tree_size = len(self._start_tree['nodes'])
        combined_parents = list(self._start_tree['parents'])
        for parent_idx in self._goal_tree['parents']:
            if parent_idx == -1:
                combined_parents.append(-1)
            else:
                combined_parents.append(parent_idx + start_tree_size)
        
        if path_found:
            # Extract path from start tree
            path_from_start = []
            # Use the stored index instead of looking it up
            current_idx = connecting_idx_a 
            while current_idx != -1:
                path_from_start.insert(0, self._start_tree['nodes'][current_idx])
                current_idx = self._start_tree['parents'][current_idx]
            
            # Extract path from goal tree
            path_from_goal = []
            # Use the stored index instead of looking it up
            current_idx = connecting_idx_b
            while current_idx != -1:
                path_from_goal.append(self._goal_tree['nodes'][current_idx])
                current_idx = self._goal_tree['parents'][current_idx]
            
            # Combine paths
            raw_path = path_from_start + path_from_goal
        else:
            print(f"RRT-Connect: Failed to find path after {self.max_iterations} iterations.")
            return [], combined_nodes, combined_parents
        
        # --- Smoothing (Conditional) ---
        if path_found and smooth_path:
            smoothed_path = self._smoother.smooth(raw_path, self._is_segment_collision_free)
            return smoothed_path, combined_nodes, combined_parents
        else:
            return raw_path, combined_nodes, combined_parents
    
    def _extend(self, tree: dict, target_point: Tuple[float, float]) -> int:
        """
        Extends the tree towards the target point.
        
        Args:
            tree: The tree to extend (either start_tree or goal_tree)
            target_point: The point to extend towards
            
        Returns:
            Index of the new node in the tree, or -1 if extension failed
        """
        # Find the nearest node in the tree
        nearest_idx = self._find_nearest(tree['nodes'], target_point)
        nearest_node = tree['nodes'][nearest_idx]
        
        # Steer towards the target
        new_node = self._steer(nearest_node, target_point, self.step_size)
        
        # Check if the new node is valid
        if self._is_segment_collision_free(nearest_node, new_node):
            # Check bounds
            if not (self.robot_radius <= new_node[0] <= self.width - self.robot_radius and
                    self.robot_radius <= new_node[1] <= self.height - self.robot_radius):
                return -1  # Out of bounds
            
            # Add new node to the tree
            tree['nodes'].append(new_node)
            tree['parents'].append(nearest_idx)
            return len(tree['nodes']) - 1
        
        return -1  # Extension failed due to collision
    
    def _connect(self, tree: dict, target_point: Tuple[float, float]) -> int:
        """
        Repeatedly extends the tree towards the target point until it
        reaches the target or encounters an obstacle.
        
        Args:
            tree: The tree to connect
            target_point: The point to connect to
            
        Returns:
            Index of the last valid node added to the tree, or -1 if connection failed
        """
        new_node_idx = -1
        
        while True:
            extend_result = self._extend(tree, target_point)
            
            if extend_result == -1:
                # Extension failed
                return new_node_idx
                
            new_node_idx = extend_result
            new_node = tree['nodes'][new_node_idx]
            
            # Check if we've reached the target (allowing for small numerical differences)
            if np.linalg.norm(np.array(new_node) - np.array(target_point)) < 1e-6:
                return new_node_idx
    
    def _find_nearest(self, nodes: List[Tuple[float, float]], point: Tuple[float, float]) -> int:
        """Finds the index of the node in the list closest to the point."""
        nodes_arr = np.array(nodes)
        point_arr = np.array(point)
        distances_sq = np.sum((nodes_arr - point_arr)**2, axis=1)
        return np.argmin(distances_sq)

    def _steer(self,
               from_node: Tuple[float, float],
               to_node: Tuple[float, float],
               step_size: float
               ) -> Tuple[float, float]:
        """Steers from 'from_node' towards 'to_node' by 'step_size'."""
        from_arr = np.array(from_node)
        to_arr = np.array(to_node)
        vec = to_arr - from_arr
        dist = np.linalg.norm(vec)

        if dist < 1e-9:  # Avoid division by zero if nodes are identical
            return tuple(from_arr)

        if dist <= step_size:
            # Target is within reach
            return tuple(to_arr)
        else:
            # Move step_size towards target
            unit_vec = vec / dist
            new_node_arr = from_arr + unit_vec * step_size
            return tuple(new_node_arr)

    def _is_segment_collision_free(self,
                                   from_node: Tuple[float, float],
                                   to_node: Tuple[float, float]
                                   ) -> bool:
        """
        Checks if the segment between two nodes collides with planning obstacles
        or goes out of bounds.
        This method is also used by the PathSmoother via dependency injection.
        """
        # Check node bounds first (important for smoother too)
        for node in [from_node, to_node]:
            x, y = node
            if not (self.robot_radius <= x <= self.width - self.robot_radius and
                    self.robot_radius <= y <= self.height - self.robot_radius):
                return False  # Node itself is out of bounds

        # Check line segment against obstacles
        for obstacle in self._planning_obstacles:
            # Use the obstacle's intersects_segment method
            # Pass the current robot_radius for the check
            if obstacle.intersects_segment(from_node, to_node, self.robot_radius):
                return False  # Collision detected

        return True  # Segment is collision-free
