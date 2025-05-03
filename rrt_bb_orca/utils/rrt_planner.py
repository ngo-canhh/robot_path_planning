import numpy as np
from typing import List, Tuple, Callable # Added typing for clarity

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
                    # print(f"Smoothing: Shortcut from index {current_idx} to {best_next_idx}")
                    break # Take the furthest valid shortcut

            smoothed_path.append(path[best_next_idx])
            current_idx = best_next_idx # Jump to the node we connected to

        # print(f"Smoothing: Original length {len(path)}, Smoothed length {len(smoothed_path)}")
        return smoothed_path



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
        self.goal = None

        # Internal state reset during planning
        self._nodes: List[Tuple[float, float]] = []
        self._parents: List[int] = []
        self._planning_obstacles: list = [] # List of Obstacle objects used in the current plan

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
                  smooth_path: bool = False # Added option to control smoothing
                 ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[int]]:
        """
        Plan path using RRT, optionally smoothing the result.

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
            - final_nodes: List of all nodes explored by RRT.
            - final_parents: List of parent indices for each node in final_nodes.
        """
        # --- Reset internal state for new planning ---
        self._planning_obstacles = obstacles_for_planning
        self._nodes = [(start_x, start_y)]
        self._parents = [-1]
        # ---

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
            nearest_idx = self._find_nearest(self._nodes, rnd_point)
            nearest_node = self._nodes[nearest_idx]

            # 3. Steer
            new_node = self._steer(nearest_node, rnd_point, self.step_size)

            # Check bounds more robustly during steering/expansion
            # if not (self.robot_radius <= new_node[0] <= self.width - self.robot_radius and
            #         self.robot_radius <= new_node[1] <= self.height - self.robot_radius):
            #      continue # Skip if the target steered point is out

            # 4. Check collision for the new segment
            if self._is_segment_collision_free(nearest_node, new_node):
                # Check bounds *after* ensuring the path is clear
                # This prevents adding nodes slightly outside bounds if the path check passes
                # but the final point is edge-case outside. Re-check node bounds strictly.
                if not (self.robot_radius <= new_node[0] <= self.width - self.robot_radius and
                        self.robot_radius <= new_node[1] <= self.height - self.robot_radius):
                    continue # Skip node if it lands out of bounds

                # 5. Add node and edge
                self._nodes.append(new_node)
                self._parents.append(nearest_idx)
                new_node_idx = len(self._nodes) - 1

                # 6. Check goal reached
                dist_to_goal = np.linalg.norm(np.array(new_node) - np.array((goal_x, goal_y)))
                if dist_to_goal <= self.min_dist_to_goal:
                     # Try connecting the new node directly to the goal
                     if self._is_segment_collision_free(new_node, (goal_x, goal_y)):
                          # Ensure goal itself is within bounds (usually guaranteed by problem setup)
                          if (self.robot_radius <= goal_x <= self.width - self.robot_radius and
                               self.robot_radius <= goal_y <= self.height - self.robot_radius):
                                self._nodes.append((goal_x, goal_y))
                                self._parents.append(new_node_idx)
                                goal_node_idx = len(self._nodes) - 1
                                print(f"RRT: Path found connecting near goal node in {i+1} iterations.")
                                path_found = True
                                break
                          # else: Goal is out of bounds, cannot connect


        # --- Path Extraction ---
        raw_path = []
        # Make copies to return the state of the tree at the end
        final_nodes = list(self._nodes)
        final_parents = list(self._parents)

        if path_found:
            current_idx = goal_node_idx
            while current_idx != -1:
                raw_path.insert(0, self._nodes[current_idx])
                # Ensure parent index is valid before accessing
                if current_idx < len(self._parents):
                    current_idx = self._parents[current_idx]
                else:
                    # Should not happen in correct RRT logic, but safety check
                    print("Error: Invalid parent index during path reconstruction.")
                    raw_path = [] # Invalidate path
                    break
        else:
            print(f"RRT: Failed to find path after {self.max_iterations} iterations.")
            # Return empty path but still return the explored tree
            return [], final_nodes, final_parents

        # --- Smoothing (Conditional) ---
        if path_found and smooth_path:
             # Pass the collision check method of this planner instance
             # The smoother uses this specific instance's method and obstacles
             smoothed_path = self._smoother.smooth(raw_path, self._is_segment_collision_free)
             return smoothed_path, final_nodes, final_parents
        else:
             # Return the raw path if smoothing is disabled or path not found
             return raw_path, final_nodes, final_parents


    def _find_nearest(self, nodes: List[Tuple[float, float]], point: Tuple[float, float]) -> int:
        """Finds the index of the node in the list closest to the point."""
        # Optimization: Avoid recreating np.array if nodes list is huge and called often
        # For typical RRT sizes, this is fine.
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

        if dist < 1e-9: # Avoid division by zero if nodes are identical
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
        This method is now also used by the PathSmoother via dependency injection.
        """
        # Check node bounds first (important for smoother too)
        for node in [from_node, to_node]:
            x, y = node
            if not (self.robot_radius <= x <= self.width - self.robot_radius and
                    self.robot_radius <= y <= self.height - self.robot_radius):
                 # print(f"Segment check fail: Node {node} out of bounds")
                 return False # Node itself is out of bounds

        # Check line segment against obstacles
        for obstacle in self._planning_obstacles:
            # Use the obstacle's intersects_segment method
            # Pass the current robot_radius for the check
            if obstacle.intersects_segment(from_node, to_node, self.robot_radius):
                 # print(f"Segment check fail: Collision with obs at ({obstacle.x:.1f},{obstacle.y:.1f}) for segment {from_node}->{to_node}")
                 return False # Collision detected

        # print(f"Segment check OK: {from_node} -> {to_node}")
        return True # Segment is collision-free