import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from queue import PriorityQueue
import math
from scipy.interpolate import splprep, splev
from components.obstacle import StaticObstacle, DynamicObstacle

class OraclePathPlanner:
    """
    A path planner that uses A* algorithm with grid decomposition to find an oracle path.
    This path is used to calculate the optimal path length and smoothness metrics.
    All dynamic obstacles are treated as static for this calculation.
    """
    
    def __init__(self, width, height, robot_radius, grid_size=5):
        """
        Initialize the Oracle Path Planner.
        
        Args:
            width (int): The width of the environment.
            height (int): The height of the environment.
            robot_radius (float): The radius of the robot.
            grid_size (int): The size of each grid cell.
        """
        self.width = width
        self.height = height
        self.robot_radius = robot_radius
        self.grid_size = grid_size
        
        # Calculate grid dimensions
        self.grid_width = int(np.ceil(width / grid_size))
        self.grid_height = int(np.ceil(height / grid_size))
        
        # Initialize grid
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        # Path storage
        self.raw_path = None  # A* path before smoothing
        self.smoothed_path = None  # Path after smoothing
        
    def reset_grid(self):
        """Reset the grid to all free cells."""
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
    def mark_obstacle(self, obstacle):
        """
        Mark an obstacle on the grid.
        
        Args:
            obstacle: An Obstacle object (either StaticObstacle or DynamicObstacle).
        """
        # Get the obstacle's shape
        shape = obstacle.shape
        
        # Calculate the bounding box of the obstacle based on shape type
        min_x, min_y, max_x, max_y = None, None, None, None
        
        if hasattr(shape, 'get_bounding_box'):
            # Use get_bounding_box if available
            min_x, min_y, max_x, max_y = shape.get_bounding_box(obstacle.x, obstacle.y)
        else:
            # Manual calculation for different shape types
            effective_radius = shape.get_effective_radius()
            
            # Calculate simple bounding box using effective radius
            min_x = obstacle.x - effective_radius
            min_y = obstacle.y - effective_radius
            max_x = obstacle.x + effective_radius
            max_y = obstacle.y + effective_radius
            
            # For rectangles, we can be more precise
            if hasattr(shape, 'width') and hasattr(shape, 'height'):
                # This is probably a rectangle
                # Get corners in world coordinates
                if hasattr(shape, '_get_world_corners'):
                    corners = shape._get_world_corners(obstacle.x, obstacle.y)
                    x_coords = [corner[0] for corner in corners]
                    y_coords = [corner[1] for corner in corners]
                    min_x = min(x_coords)
                    min_y = min(y_coords)
                    max_x = max(x_coords)
                    max_y = max(y_coords)
        
        # Expand bounding box by robot radius
        min_x -= self.robot_radius
        min_y -= self.robot_radius
        max_x += self.robot_radius
        max_y += self.robot_radius
        
        # Convert to grid coordinates
        grid_min_x = max(0, int(min_x / self.grid_size))
        grid_min_y = max(0, int(min_y / self.grid_size))
        grid_max_x = min(self.grid_width - 1, int(np.ceil(max_x / self.grid_size)))
        grid_max_y = min(self.grid_height - 1, int(np.ceil(max_y / self.grid_size)))
        
        # Check each grid cell to see if it intersects with the obstacle
        for grid_y in range(grid_min_y, grid_max_y + 1):
            for grid_x in range(grid_min_x, grid_max_x + 1):
                # Get the world coordinates of the cell center
                cell_center_x = (grid_x + 0.5) * self.grid_size
                cell_center_y = (grid_y + 0.5) * self.grid_size
                
                # Check if the cell intersects with the obstacle (including robot radius)
                if obstacle.check_collision(cell_center_x, cell_center_y, self.robot_radius):
                    self.grid[grid_y, grid_x] = True  # Mark as occupied
    
    def build_grid_map(self, obstacles):
        """
        Build a grid map from the given obstacles.
        
        Args:
            obstacles: List of Obstacle objects.
        """
        # Reset grid
        self.reset_grid()
        
        # Mark each obstacle on the grid
        for obstacle in obstacles:
            self.mark_obstacle(obstacle)
            
    def heuristic(self, a, b):
        """
        Calculate the heuristic value (Euclidean distance).
        
        Args:
            a: Grid coordinates (x, y).
            b: Grid coordinates (x, y).
            
        Returns:
            float: Euclidean distance between a and b.
        """
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        
    def a_star_search(self, start, goal):
        """
        Perform A* search on the grid.
        
        Args:
            start: Start position in world coordinates (x, y).
            goal: Goal position in world coordinates (x, y).
            
        Returns:
            list: List of points (x, y) forming the path in world coordinates, or None if no path is found.
        """
        # Convert start and goal to grid coordinates
        start_grid = (int(start[0] / self.grid_size), int(start[1] / self.grid_size))
        goal_grid = (int(goal[0] / self.grid_size), int(goal[1] / self.grid_size))
        
        # Ensure start and goal are within grid bounds
        start_grid = (
            max(0, min(start_grid[0], self.grid_width - 1)),
            max(0, min(start_grid[1], self.grid_height - 1))
        )
        goal_grid = (
            max(0, min(goal_grid[0], self.grid_width - 1)),
            max(0, min(goal_grid[1], self.grid_height - 1))
        )
        
        # Check if start or goal is in obstacle
        if self.grid[start_grid[1], start_grid[0]] or self.grid[goal_grid[1], goal_grid[0]]:
            print("Warning: Start or goal is inside an obstacle in the grid.")
            # Try to find nearest free cell for start/goal
            if self.grid[start_grid[1], start_grid[0]]:
                start_grid = self.find_nearest_free_cell(start_grid)
                if start_grid is None:
                    print("Error: Cannot find a free cell near start.")
                    return None
            if self.grid[goal_grid[1], goal_grid[0]]:
                goal_grid = self.find_nearest_free_cell(goal_grid)
                if goal_grid is None:
                    print("Error: Cannot find a free cell near goal.")
                    return None
        
        # Initialize open set (priority queue)
        open_set = PriorityQueue()
        open_set.put((0, start_grid))
        
        # Initialize came_from and cost_so_far dictionaries
        came_from = {}
        cost_so_far = {start_grid: 0}
        
        # Possible movements (8-connected grid)
        movements = [
            (1, 0), (0, 1), (-1, 0), (0, -1),  # 4-connected
            (1, 1), (-1, 1), (-1, -1), (1, -1)  # Diagonals
        ]
        
        while not open_set.empty():
            # Get the node with the lowest f-score
            _, current = open_set.get()
            
            # If reached the goal, reconstruct path
            if current == goal_grid:
                path = []
                while current in came_from:
                    # Convert grid coordinates to world coordinates (center of cell)
                    world_x = (current[0] + 0.5) * self.grid_size
                    world_y = (current[1] + 0.5) * self.grid_size
                    path.append((world_x, world_y))
                    current = came_from[current]
                    
                # Add start point (converted to world coordinates)
                world_start_x = (start_grid[0] + 0.5) * self.grid_size
                world_start_y = (start_grid[1] + 0.5) * self.grid_size
                path.append((world_start_x, world_start_y))
                
                # Reverse path (from start to goal)
                path.reverse()
                
                # Add actual goal point at the end
                path.append(goal)
                
                return path
            
            # Explore neighbors
            for dx, dy in movements:
                next_x = current[0] + dx
                next_y = current[1] + dy
                
                # Check if next node is within grid bounds
                if not (0 <= next_x < self.grid_width and 0 <= next_y < self.grid_height):
                    continue
                    
                # Check if next node is obstacle-free
                if self.grid[next_y, next_x]:
                    continue
                
                # Calculate new cost (diagonal moves cost √2)
                if abs(dx) + abs(dy) == 2:  # Diagonal
                    new_cost = cost_so_far[current] + 1.414  # √2
                else:
                    new_cost = cost_so_far[current] + 1
                
                next_node = (next_x, next_y)
                
                # If new path is better
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.heuristic(next_node, goal_grid)
                    open_set.put((priority, next_node))
                    came_from[next_node] = current
        
        # No path found
        print("No path found from start to goal.")
        return None
    
    def find_nearest_free_cell(self, cell):
        """
        Find the nearest free cell to the given cell.
        
        Args:
            cell: Grid coordinates (x, y).
            
        Returns:
            tuple: Coordinates of the nearest free cell, or None if not found.
        """
        # BFS to find nearest free cell
        visited = set()
        queue = [(cell, 0)]  # (cell, distance)
        visited.add(cell)
        
        max_distance = 20  # Limit search radius
        
        while queue:
            (x, y), dist = queue.pop(0)
            
            # If this cell is free, return it
            if not self.grid[y, x]:
                return (x, y)
                
            # If reached max distance, stop searching
            if dist >= max_distance:
                continue
                
            # Check neighbors
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                    continue
                    
                next_cell = (nx, ny)
                if next_cell not in visited:
                    visited.add(next_cell)
                    queue.append((next_cell, dist + 1))
                    
        return None  # No free cell found
    
    def smooth_path(self, path, smoothness=0.2, num_points=100):
        """
        Smooth the path using cubic spline interpolation.
        
        Args:
            path: List of points (x, y) forming the path.
            smoothness: Smoothing factor (0 to 1).
            num_points: Number of points to generate in the smoothed path.
            
        Returns:
            list: Smoothed path as a list of points (x, y).
        """
        if path is None or len(path) < 2:
            return path
            
        # Extract x and y coordinates
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        
        # Generate spline
        if len(path) < 4:
            # Not enough points for a spline, use linear interpolation
            t = np.linspace(0, 1, num_points)
            x_smooth = np.interp(t, np.linspace(0, 1, len(path)), x)
            y_smooth = np.interp(t, np.linspace(0, 1, len(path)), y)
            smoothed_path = list(zip(x_smooth, y_smooth))
        else:
            try:
                # Fit spline with smoothing
                tck, u = splprep([x, y], s=smoothness * len(path), k=min(3, len(path)-1))
                
                # Generate new points along the spline
                u_new = np.linspace(0, 1, num_points)
                x_smooth, y_smooth = splev(u_new, tck)
                
                # Create smoothed path
                smoothed_path = list(zip(x_smooth, y_smooth))
            except Exception as e:
                print(f"Warning: Spline smoothing failed: {e}")
                # Fall back to linear interpolation
                t = np.linspace(0, 1, num_points)
                x_smooth = np.interp(t, np.linspace(0, 1, len(path)), x)
                y_smooth = np.interp(t, np.linspace(0, 1, len(path)), y)
                smoothed_path = list(zip(x_smooth, y_smooth))
        
        return smoothed_path
    
    def plan_path(self, start, goal, obstacles):
        """
        Plan a path from start to goal considering the given obstacles.
        
        Args:
            start: Start position in world coordinates (x, y).
            goal: Goal position in world coordinates (x, y).
            obstacles: List of Obstacle objects.
            
        Returns:
            tuple: (raw_path, smoothed_path, path_length, path_smoothness)
        """
        # Build grid map
        self.build_grid_map(obstacles)
        
        # Find raw path using A*
        raw_path = self.a_star_search(start, goal)
        self.raw_path = raw_path
        
        if raw_path is None:
            print("No path found.")
            return None, None, float('inf'), float('inf')
        
        # Smooth the path
        smoothed_path = self.smooth_path(raw_path)
        self.smoothed_path = smoothed_path
        
        # Calculate path length
        path_length = self.calculate_path_length(smoothed_path)
        
        # Calculate path smoothness
        path_smoothness = self.calculate_path_smoothness(smoothed_path)
        
        return raw_path, smoothed_path, path_length, path_smoothness
    
    def calculate_path_length(self, path):
        """
        Calculate the total length of the path.
        
        Args:
            path: List of points (x, y) forming the path.
            
        Returns:
            float: Path length.
        """
        if path is None or len(path) < 2:
            return float('inf')
            
        length = 0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
        return length
    
    def calculate_path_smoothness(self, path):
        """
        Calculate the smoothness of the path as the average angle between vectors formed by points
        with a window spacing, rather than consecutive points. This better captures sharp turns.
        
        Args:
            path: List of points (x, y) forming the path.
            
        Returns:
            float: Path smoothness (average angle in radians).
        """
        if path is None or len(path) < 3:
            return 0.0
            
        # Define window size (adaptive based on path length)
        window_size = max(1, min(10, len(path) // 10))
        
        angles = []
        for i in range(window_size, len(path) - window_size):
            # Use windowed points instead of consecutive points
            p_prev = np.array(path[i - window_size])
            p_curr = np.array(path[i])
            p_next = np.array(path[i + window_size])
            
            # Calculate vectors
            vec1 = p_curr - p_prev
            vec2 = p_next - p_curr
            
            # Calculate magnitudes
            mag1 = np.linalg.norm(vec1)
            mag2 = np.linalg.norm(vec2)
            
            if mag1 > 1e-6 and mag2 > 1e-6:
                # Calculate dot product
                dot_product = np.dot(vec1, vec2) / (mag1 * mag2)
                dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure within valid range
                
                # Calculate angle
                angle = np.arccos(dot_product)
                angles.append(angle)
        
        if not angles:
            return 0.0
            
        return np.mean(angles)
    
    def visualize(self, start, goal, obstacles, ax=None):
        """
        Visualize the grid map, obstacles, and planned path.
        
        Args:
            start: Start position in world coordinates (x, y).
            goal: Goal position in world coordinates (x, y).
            obstacles: List of Obstacle objects.
            ax: Matplotlib axes for plotting (optional).
            
        Returns:
            matplotlib.axes.Axes: The axes with the visualization.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot grid
        grid_img = np.zeros((self.grid_height, self.grid_width, 4))
        grid_img[self.grid] = [1, 0, 0, 0.3]  # Red for obstacles
        ax.imshow(grid_img, origin='lower', extent=[0, self.width, 0, self.height], alpha=0.3)
        
        # Plot grid lines
        for i in range(0, self.width + 1, self.grid_size):
            ax.axvline(i, color='gray', linestyle='-', alpha=0.2)
        for i in range(0, self.height + 1, self.grid_size):
            ax.axhline(i, color='gray', linestyle='-', alpha=0.2)
        
        # Plot obstacles
        for obstacle in obstacles:
            patch = obstacle.get_render_patch(alpha=0.6, zorder=3)
            ax.add_patch(patch)
        
        # Plot start and goal
        ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
        ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
        
        # Plot raw path
        if self.raw_path:
            x = [p[0] for p in self.raw_path]
            y = [p[1] for p in self.raw_path]
            ax.plot(x, y, 'b-', linewidth=1, alpha=0.5, label='A* Path')
        
        # Plot smoothed path
        if self.smoothed_path:
            x = [p[0] for p in self.smoothed_path]
            y = [p[1] for p in self.smoothed_path]
            ax.plot(x, y, 'g-', linewidth=2, label='Smoothed Path')
        
        # Set axes limits and labels
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Oracle Path Planning with A* and Grid Decomposition')
        ax.legend()
        
        return ax 