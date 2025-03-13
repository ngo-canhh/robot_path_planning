import matplotlib.pyplot as plt
from shape import Circle, Rectangle  # Import Shape classes
from obstacle import StaticObstacle, DynamicObstacle # Import Obstacle classes
from ray_tracing import RayTracing # Import RayTracing
from waiting_rule import WaitingRule # Import WaitingRule
from rrt_planner import RRTPlanner # Import RRTPlanner

show_animation = True # Flag to enable/disable animation

def main(gx=55.0, gy=5.0, random_seed=None):
    print("start " + __file__)

    # Creating the obstacles to match the image
    static_obstacles = [
        # Top-left rectangle (tall)
          # Top-left rectangle (short)
        StaticObstacle(Rectangle(20, 5, 3, 5)),
        
        # Oval/Circle in the middle-left
        StaticObstacle(Circle(15, 30, 4)),
        
        # Triangle at top-right (approximated with a rectangle)
        StaticObstacle(Rectangle(35, 5, 7, 7)),
        
        # Cross/plus sign at top-right
        StaticObstacle(Rectangle(30, 25, 10, 2)),  # Horizontal part
        StaticObstacle(Rectangle(34, 21, 2, 10)) , # Vertical part        
        # Hexagon at bottom-left (approximated with a circle)
        StaticObstacle(Rectangle(8, 47, 6, 10)),
        
        # Pentagon at bottom-right (approximated with a circle)
        StaticObstacle(Circle(43, 45, 4)),
    ]
    dynamic_obstacles = [
        # DynamicObstacle(Rectangle(30, 25, 10, 2), -1,-1),  # Horizontal part
        # DynamicObstacle(Rectangle(34, 21, 2, 10), -1,-1) , 
    ]
    
    obstacle_list = static_obstacles + dynamic_obstacles;  # No dynamic obstacles as per the image

    # Initialize RayTracing and WaitingRule
    ray_tracer = RayTracing(static_obstacles)
    waiting_rule = WaitingRule(robot_speed=1.0)

    # Set Initial parameters - start at top-left, goal at bottom-right
    rrt_planner = RRTPlanner(
        start=[0, 55],           # Top-left corner
        goal=[gx, gy],          # Bottom-right corner
        rand_area=[-1, 60],
        obstacle_list=obstacle_list,
        robot_radius=0.8,
        play_area=[-5.0, 60.0, -5.0, 60.0],
        path_resolution=1,
        ray_tracer=ray_tracer,
        waiting_rule=waiting_rule,
        random_seed=random_seed,
    )
    path = rrt_planner.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            rrt_planner.draw_graph()
            # Add path length text to the figure
            plt.text(30, 57, f"Path Length: {rrt_planner.path_length:.2f}", 
                     bbox=dict(facecolor='white', alpha=0.7))
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            # Draw a line to show the environment division as in the image
            plt.plot([0, 60], [30, 30], 'b-', linewidth=1)
            plt.grid(True)
            plt.pause(0.01)
            plt.show()


if __name__ == '__main__':
    main(random_seed=10)  # Using a single run with a fixed seed