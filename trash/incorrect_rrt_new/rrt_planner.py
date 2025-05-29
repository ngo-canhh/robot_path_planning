import random
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # Add this import
from matplotlib.offsetbox import OffsetImage, AnnotationBbox  # Add this import
import os.path as path

import numpy as np

from shape import Circle, Rectangle  # Import Shape classes
from obstacle import StaticObstacle, DynamicObstacle # Import obstacle classes
from ray_tracing import RayTracing # Import RayTracing
from waiting_rule import WaitingRule # Import WaitingRule
from smoother import Smoother # Import Smoother



show_animation = True


class RRTPlanner:
    """
    Class for RRT path planning algorithm (algorithm module - visualization separated).
    """

    class Node:
        """
        RRT Node class.
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    class AreaBounds:
        """
        Class to define area bounds.
        """
        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])


    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=500,
                 play_area=None,
                 robot_radius=0.0,
                 ray_tracer=None, # RayTracing object
                 waiting_rule=None, # WaitingRule object
                 random_seed=None, # Random seed for reproducibility
                 smoother: Smoother = None, # Smoothing object
                 show_animation=False
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [Obstacle objects]
        randArea:Random Sampling Area [min,max]
        play_area:stay inside this area [xmin,xmax,ymin,ymax]
        robot_radius: robot body modeled as circle with given radius
        ray_tracer: RayTracing object for static obstacle avoidance
        waiting_rule: WaitingRule object for dynamic obstacle avoidance
        visualizer: RRTVisualizer object for handling visualization

        """
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius
        self.ray_tracer = ray_tracer # RayTracing object
        self.waiting_rule = waiting_rule # WaitingRule object
        self.simulation_time = 0.0  # Initialize simulation time
        self.path_length = 0.0  # Initialize path length
        self.planner_path = None  # Initialize planner path
        self.final_path = None  # Initialize final path
        self.smoother = smoother  # Initialize smoothing path
        if show_animation:
          self.visualizer = RRTVisualizer(self) # RRTVisualizer object


    def planning(self, animation=True):
        """
        rrt path planning
        """
        self.simulation_time = 0.0  # Reset simulation time
        self.node_list = [self.start]
        for i in range(self.max_iter):
            # Update dynamic obstacles' positions
            self.update_dynamic_obstacles(dt=0.1)  # dt is the time step, adjust as needed

            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_if_outside_play_area(new_node, self.play_area) and \
            self.check_collision_free(new_node, self.obstacle_list, new_node):
                self.node_list.append(new_node)

            if animation and i % 5 == 0 and self.visualizer: # Use visualizer to draw
                self.visualizer.draw_graph(rnd_node=rnd_node, rrt_planner=self)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision_free(final_node, self.obstacle_list, final_node):
                    self.generate_final_course(len(self.node_list) - 1)
                    return self.final_path

            if animation and i % 5 and self.visualizer: # Use visualizer to draw
                self.visualizer.draw_graph(rnd_node=rnd_node, rrt_planner=self)

        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        Steer from from_node towards to_node.
        """
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node

    def calc_path_length(self, path):
        """
        Calculate the total length of a path.

        Args:
            path: List of [x, y] coordinates

        Returns:
            float: Total path length
        """
        length = 0.0
        for i in range(len(path) - 1):
            dx = path[i][0] - path[i+1][0]
            dy = path[i][1] - path[i+1][1]
            length += math.hypot(dx, dy)
        return length

    def generate_final_course(self, goal_ind):
        """
        Generate final path from goal node to start node.
        """
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        self.planner_path = path
        if self.smoother:
            self.final_path = self.smoother.smooth_path(path)
        else:
            self.final_path = path

        # Calculate and store the path length
        self.path_length = self.calc_path_length(self.final_path)

        return self.final_path

    def calc_dist_to_goal(self, x, y):
        """
        Calculate distance to goal position.
        """
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        """
        Get random node for RRT expansion.
        """
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd


    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        """Get nearest node index from RRT tree to random node."""
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):
        """Check if node is outside play area."""
        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
           node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    def check_collision_free(self, node, obstacleList, check_node):
        """
        Check collision-free for a new node with obstacles.
        Integrates Ray Tracing for static and (partially) Waiting Rule for dynamic obstacles.
        """
        if node is None:
            return False

        for obstacle in obstacleList:
            if obstacle.get_min_distance(node.x, node.y) <= self.robot_radius: # Basic collision check using min_distance
                return False

            if isinstance(obstacle, StaticObstacle) and self.ray_tracer: # Ray Tracing for static obstacles
                parent_node = node.parent
                if parent_node:
                    if self.ray_tracer.check_ray_collision((parent_node.x, parent_node.y), (node.x, node.y)):
                        return False # Ray between parent and new node intersects static obstacle

            elif isinstance(obstacle, DynamicObstacle) and self.waiting_rule: # Dynamic obstacle handling (Waiting Rule - IMPROVEMENT NEEDED)
                # **WAITING RULE INTEGRATION NEEDED HERE - CURRENTLY JUST A PLACEHOLDER**
                # Placeholder: Simple distance check for dynamic obstacles - REPLACE WITH WAITING RULE LOGIC
                safe_distance_dynamic = 2.0 # Example safe distance for dynamic obstacles
                if not self.waiting_rule.is_safe_distance((node.x, node.y), obstacle, safe_distance_dynamic):
                    waiting_time = self.waiting_rule.calculate_waiting_time( # Calculate waiting time (WIP)
                        robot_pos=(node.x, node.y),
                        robot_orientation=self.get_node_orientation(check_node), # Pass check_node for orientation
                        dynamic_obstacle=obstacle
                    )
                    if waiting_time > 0:
                        print(f"Dynamic obstacle nearby, waiting needed: {waiting_time:.2f} s (not actually waiting in code)")
                        return False # Treat as collision if waiting is needed (placeholder)
                        # **IMPLEMENT ACTUAL WAITING OR PATH ADJUSTMENT BASED ON WAITING TIME**

        return True  # Safe, no collision

    def get_node_orientation(self, node):
        """Get node orientation based on parent node."""
        if node.parent:
            return math.atan2(node.y - node.parent.y, node.x - node.parent.x)
        return 0.0 # Default orientation if no parent

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        """Calculate distance and angle between two nodes."""
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def update_dynamic_obstacles(self, dt):
        """
        Update positions of dynamic obstacles based on their velocity and time step dt.
        """
        self.simulation_time += dt  # Track simulation time
        for obstacle in self.obstacle_list:
            if isinstance(obstacle, DynamicObstacle):
                # Update position based on velocity and time step
                obstacle.move(dt)


class RRTStarPlanner(RRTPlanner):
    """
    RRT* Planner class with path optimization
    """

    class Node(RRTPlanner.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0  # Add cost attribute

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connect_circle_dist = 50.0  # Increased search radius
        self.start.cost = 0.0  # Initialize start node cost

    def planning(self, animation=True):
        """
        RRT* planning algorithm with path optimization
        """
        self.node_list = [self.start]
        for i in range(self.max_iter):
            # Update dynamic obstacles' positions
            self.update_dynamic_obstacles(dt=0.1)

            # Random sampling with bias
            rnd_node = self.get_random_node()
            
            # Find nearest node
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            # Steer towards random node
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)
            
            if new_node is None:
                continue

            new_node.cost = nearest_node.cost + self.calc_distance(nearest_node, new_node)

            # Collision check
            if self.check_collision_free(new_node, self.obstacle_list, new_node):
                # Find nearby nodes within connection radius
                neighbor_indices = self.find_near_nodes(new_node)
                
                # Choose optimal parent
                min_cost_node = self.choose_parent(new_node, neighbor_indices)
                
                # Add to tree
                if min_cost_node:
                    self.node_list.append(min_cost_node)
                    
                    # Rewire the tree
                    self.rewire(min_cost_node, neighbor_indices)

            # Visualization and goal check
            if animation and i % 5 == 0 and self.visualizer:
                self.visualizer.draw_graph(rnd_node=rnd_node, rrt_planner=self)

            # Goal check with more lenient condition
            for node in self.node_list:
                if self.calc_dist_to_goal(node.x, node.y) <= self.expand_dis:
                    final_node = self.steer(node, self.end, self.expand_dis)
                    if final_node and self.check_collision_free(final_node, self.obstacle_list, final_node):
                        self.generate_final_course(self.node_list.index(node))
                        return self.final_path

        return None

    def calc_distance(self, from_node, to_node):
        return math.hypot(from_node.x - to_node.x, from_node.y - to_node.y)

    def find_near_nodes(self, new_node):
        """
        Find nodes within connection radius
        """
        n = len(self.node_list) + 1
        # Adjust radius calculation for larger search space
        r = self.connect_circle_dist * math.sqrt(math.log(n) / n)
        
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2
                    for node in self.node_list]
        return [i for i, d in enumerate(dist_list) if d <= r**2]

    def choose_parent(self, new_node, neighbor_indices):
        """
        Choose parent with minimum cost
        """
        if not neighbor_indices:
            return new_node

        # Try to connect to the nearest node first
        if neighbor_indices:
            best_cost = float('inf')
            best_parent = None

            for i in neighbor_indices:
                near_node = self.node_list[i]
                d = self.calc_distance(near_node, new_node)
                
                # Check if connecting through this node reduces overall cost
                tentative_cost = near_node.cost + d
                
                if tentative_cost < best_cost and \
                   self.check_collision(near_node, new_node):
                    best_cost = tentative_cost
                    best_parent = near_node

            if best_parent:
                new_node.parent = best_parent
                new_node.cost = best_cost
                return new_node

        return new_node

    def rewire(self, new_node, neighbor_indices):
        """
        Rewire the tree to optimize paths
        """
        for i in neighbor_indices:
            near_node = self.node_list[i]
            d = self.calc_distance(new_node, near_node)
            
            # Potential new cost if near_node connects through new_node
            new_cost = new_node.cost + d

            # Check if this new path is more efficient
            if new_cost < near_node.cost and \
               self.check_collision(new_node, near_node):
                # Update the parent and propagate cost changes
                near_node.parent = new_node
                near_node.cost = new_cost
                self.propagate_cost_to_leaves(near_node)

    def propagate_cost_to_leaves(self, parent_node):
        """
        Update costs for all descendants
        """
        for node in self.node_list:
            if node.parent == parent_node:
                # Recalculate cost based on new parent
                node.cost = parent_node.cost + self.calc_distance(parent_node, node)
                # Recursively update descendants
                self.propagate_cost_to_leaves(node)

    def check_collision(self, start_node, end_node):
        """
        Check collision between two nodes
        """
        temp_node = self.Node(end_node.x, end_node.y)
        temp_node.parent = start_node
        return self.check_collision_free(temp_node, self.obstacle_list, temp_node)

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        Override steer to update path cost
        """
        new_node = super().steer(from_node, to_node, extend_length)
        if new_node:
            new_node.cost = from_node.cost + self.calc_distance(from_node, new_node)
        return new_node


class RRTVisualizer:
    """
    Class for visualizing RRT planning process.
    """
    def __init__(self, planner: RRTPlanner):
        """
        Initialize RRTVisualizer with necessary parameters for drawing.

        Args:
            play_area (RRTPlanner.AreaBounds): Play area bounds.
            start (RRTPlanner.Node): Start node.
            end (RRTPlanner.Node): End node.
            robot_radius (float): Robot radius.
            obstacle_list (list): List of obstacle objects.
        """
        self.play_area = planner.play_area
        self.start = planner.start
        self.end = planner.end
        self.robot_radius = planner.robot_radius
        self.obstacle_list = planner.obstacle_list

        self.image_cache = {}


    def draw_graph(self, rnd_node, rrt_planner):
        """
        Draw RRT graph for animation.
        """
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd_node is not None:
            plt.plot(rnd_node.x, rnd_node.y, "^k")
            if self.robot_radius > 0.0:
                self.plot_circle(rnd_node.x, rnd_node.y, self.robot_radius, color='red')
        for node in rrt_planner.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        # Draw play area boundaries first so obstacles appear on top
        if self.play_area is not None:
            plt.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.ymin, self.play_area.ymin,
                      self.play_area.ymax, self.play_area.ymax,
                      self.play_area.ymin],
                     "-k")

        # Draw obstacles
        for obstacle in self.obstacle_list:
            if hasattr(obstacle, 'image_path') and obstacle.image_path:
                # If the obstacle has an image path, display the image
                self.plot_image(obstacle)
            else:
                # Otherwise, use the original shape visualization
                if isinstance(obstacle, StaticObstacle):
                    if isinstance(obstacle.shape, Circle):
                        self.plot_circle(obstacle.shape.x, obstacle.shape.y, obstacle.shape.radius)
                    elif isinstance(obstacle.shape, Rectangle):
                        self.plot_rectangle(obstacle.shape.x, obstacle.shape.y, obstacle.shape.width, obstacle.shape.height)
                elif isinstance(obstacle, DynamicObstacle):
                    if isinstance(obstacle.shape, Circle):
                        self.plot_circle(obstacle.shape.x, obstacle.shape.y, obstacle.shape.radius, color='blue')
                        self.plot_velocity_vector(obstacle.shape.x, obstacle.shape.y, 
                                                obstacle.vx, obstacle.vy, color='red')
                    if isinstance(obstacle.shape, Rectangle):
                        self.plot_rectangle(obstacle.shape.x, obstacle.shape.y, obstacle.shape.width, obstacle.shape.height, color='blue')
                        self.plot_velocity_vector(obstacle.shape.x + obstacle.shape.width / 2, obstacle.shape.y + obstacle.shape.height / 2, 
                                                obstacle.vx, obstacle.vy, color='red')

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        # Add simulation time display
        plt.text(5, 57,
                f"Time: {rrt_planner.simulation_time:.1f}s",
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        plt.axis("equal")
        plt.axis([self.play_area.xmin, self.play_area.xmax, self.play_area.ymin, self.play_area.ymax] if self.play_area is not None else [rrt_planner.min_rand, rrt_planner.max_rand, rrt_planner.min_rand, rrt_planner.max_rand])
        plt.grid(True)
        plt.pause(0.01)  # Ensure each frame is displayed

    def plot_image(self, obstacle):
        """Plot actual image for the obstacle with improved visibility."""
        try:
            # Get image path
            img_path = obstacle.image_path
            
            # Use cached image if available
            if img_path in self.image_cache:
                img = self.image_cache[img_path]
            else:
                # Load the image
                img = mpimg.imread(img_path)
                self.image_cache[img_path] = img
            
            # Get position and size from the obstacle
            if isinstance(obstacle.shape, Rectangle):
                x, y = obstacle.shape.x, obstacle.shape.y
                width, height = obstacle.shape.width, obstacle.shape.height
            elif isinstance(obstacle.shape, Circle):
                x, y = obstacle.shape.x - obstacle.shape.radius, obstacle.shape.y - obstacle.shape.radius
                width, height = 2 * obstacle.shape.radius, 2 * obstacle.shape.radius
            else:
                return  # Unsupported shape
            
            # Determine appropriate zoom factor - make image larger for visibility
            # Increase zoom by multiplying by a factor (e.g., 2.0)
            zoom_factor = 2.0  # Adjust this to make images larger or smaller
            img_zoom = max(width/img.shape[1], height/img.shape[0]) * zoom_factor
            
            # Create an OffsetImage with the image
            imagebox = OffsetImage(img, zoom=img_zoom)
            
            # Create and add an AnnotationBbox
            ab = AnnotationBbox(imagebox, (x + width/2, y + height/2), 
                                frameon=True,  # Add a frame around the image
                                pad=0.5,  # Add some padding
                                bboxprops=dict(edgecolor='red' if isinstance(obstacle, DynamicObstacle) else 'black'))
            plt.gca().add_artist(ab)
            
            # If it's a dynamic obstacle, add velocity vector
            if isinstance(obstacle, DynamicObstacle):
                self.plot_velocity_vector(x + width/2, y + height/2, 
                                          obstacle.vx, obstacle.vy, color='red')
                
        except Exception as e:
            print(f"Error displaying image for obstacle: {e}")
            # Fall back to regular shape visualization
            if isinstance(obstacle.shape, Circle):
                self.plot_circle(obstacle.shape.x, obstacle.shape.y, obstacle.shape.radius, 
                                 color='blue' if isinstance(obstacle, DynamicObstacle) else 'k')
            elif isinstance(obstacle.shape, Rectangle):
                self.plot_rectangle(obstacle.shape.x, obstacle.shape.y, obstacle.shape.width, obstacle.shape.height,
                                    color='blue' if isinstance(obstacle, DynamicObstacle) else 'k')

    def plot_circle(self, x, y, radius, color="k"):  # pragma: no cover
        """Plot circle for visualization."""
        circle = plt.Circle((x, y), radius,
                          color=color, fill=True, alpha=0.6)
        plt.gca().add_patch(circle) # Use color as keyword argument

    def plot_rectangle(self, x, y, width, height, color="k"):
        """Plot rectangle for visualization."""
        rect = plt.Rectangle((x, y), 
                            width, height, 
                            color=color, fill=True, alpha=0.6)
        plt.gca().add_patch(rect) # Use color as keyword argument

    def plot_velocity_vector(self, x, y, vx, vy, color="red"):
        """Plot velocity vector as an arrow."""
        # Scale the arrow length based on velocity magnitude
        scale = 1.0  # Adjust this value to change arrow length
        plt.arrow(x, y, vx * scale, vy * scale, 
                 head_width=0.6, head_length=0.8, 
                 fc=color, ec=color, alpha=0.8)