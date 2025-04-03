import matplotlib.pyplot as plt
from shape import Circle, Rectangle  # Import Shape classes
from obstacle import StaticObstacle, DynamicObstacle # Import Obstacle classes
from ray_tracing import RayTracing # Import RayTracing
from waiting_rule import WaitingRule # Import WaitingRule
from rrt_planner import RRTPlanner, RRTVisualizer # Import RRTPlanner and RRTVisualizer

show_animation = True # Flag to enable/disable animation

def main(gx=20.0, gy=45.0, random_seed=None): # Adjusted goal to be more within obstacle area for testing
    print("start " + __file__)

    # --- Original "E" shape rectangles ---
    original_e_rects = [
        # 1. Vertical Bar (left side of 'E')
        Rectangle(9, 10, 5, 40),
        # 2. Top Horizontal Bar
        Rectangle(9, 50, 20, 5),
        # 3. Middle Horizontal Bar
        Rectangle(9, 30, 20, 5),
        # 4. Bottom Horizontal Bar
        Rectangle(9, 10, 20, 5),
    ]

    mirror_e_rects = [
        # 1. Vertical Bar (left side of 'E')
        Rectangle(46, 10, 5, 40),
        # 2. Top Horizontal Bar
        Rectangle(31, 50, 20, 5),
        # 3. Middle Horizontal Bar
        Rectangle(31, 30, 20, 5),
        # 4. Bottom Horizontal Bar
        Rectangle(31, 10, 20, 5),
    ]

    # --- 1. Translated "E" to the Left ---
    translated_e_obstacles = []
    for rect in original_e_rects:
        translated_rect = Rectangle(rect.x, rect.y, rect.width, rect.height)
        translated_e_obstacles.append(StaticObstacle(translated_rect))

    # --- 2. Mirrored "E" (symmetric across x=30) ---
    mirrored_e_obstacles = []
    for rect in mirror_e_rects:
        # Calculate reflected x-coordinate: x_new = reflection_x + (reflection_x - rect.x)
        mirrored_rect = Rectangle(rect.x, rect.y, rect.width, rect.height)
        mirrored_e_obstacles.append(StaticObstacle(mirrored_rect))


    # Combine translated and mirrored E obstacles
    static_obstacles = translated_e_obstacles + mirrored_e_obstacles
    dynamic_obstacles = [] # No dynamic obstacles

    obstacle_list = static_obstacles + dynamic_obstacles;

    # Initialize RayTracing and WaitingRule
    ray_tracer = RayTracing(static_obstacles)
    waiting_rule = WaitingRule(robot_speed=1.0)


    # Set Initial parameters - start at top-left, goal more to the right
    rrt_planner = RRTPlanner(
        start=[5, 5],           # Adjusted start point
        goal=[gx, gy],          # Goal position from arguments
        rand_area=[-1, 60],
        obstacle_list=obstacle_list,
        robot_radius=0.8,
        play_area=[-5.0, 60.0, -5.0, 60.0],
        path_resolution=1,
        ray_tracer=ray_tracer,
        waiting_rule=waiting_rule,
        random_seed=random_seed,
        show_animation=True # Pass visualizer instance to RRTPlanner
    )
    path = rrt_planner.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation: # Use visualizer to draw final graph
            rrt_planner.visualizer.draw_graph(rnd_node=None, rrt_planner=rrt_planner) # No rnd_node for final draw
            # Add path length text to the figure
            plt.text(30, 57, f"Path Length: {rrt_planner.path_length:.2f}",
                     bbox=dict(facecolor='white', alpha=0.7))
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            # Draw a line to show the environment division as in the image (removed as not in this simple env)
            #plt.plot([0, 60], [30, 30], 'b-', linewidth=1) # Removed division line
            plt.grid(True)
            plt.pause(0.01)
            plt.show()


if __name__ == '__main__':
    main(random_seed=10)  # Using a single run with a fixed seed