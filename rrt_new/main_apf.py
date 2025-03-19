import matplotlib.pyplot as plt
from shape import Circle, Rectangle
from obstacle import StaticObstacle, DynamicObstacle
from ray_tracing import RayTracing
from waiting_rule import WaitingRule
from rrt_planner import RRTPlanner, RRTVisualizer
from potential_field_planner import PotentialFieldPlanner # Import PotentialFieldPlanner

show_animation = True
enable_heatmap = False # Flag to enable/disable heatmap

def main(gx=55.0, gy=5.0, random_seed=None):
    print("start " + __file__)

    AREA_WIDTH = 60.0  # Increased area width for potential field

    obs = [
        StaticObstacle(Rectangle(5, 15, 3, 10)),

        # Top-left rectangle (short)
        StaticObstacle(Rectangle(20, 5, 3, 5)),

        # Oval/Circle in the middle-left
        StaticObstacle(Circle(15, 30, 4)),

        # Triangle at top-right (approximated with a rectangle)
        StaticObstacle(Rectangle(35, 5, 7, 7)),

        # Cross/plus sign at top-right
        # StaticObstacle(Rectangle(30, 25, 10, 2)),  # Horizontal part
        # StaticObstacle(Rectangle(34, 21, 2, 10)) , # Vertical part
        # Hexagon at bottom-left (approximated with a circle)

        StaticObstacle(Rectangle(8, 47, 6, 10)),

        # Pentagon at bottom-right (approximated with a circle)
        StaticObstacle(Circle(43, 45, 4)),
        DynamicObstacle(Rectangle(30, 25, 10, 2), -1, -1),  # Horizontal part
        DynamicObstacle(Rectangle(34, 21, 2, 10), -1, -1) , # Vertical part
    ]

    obstacle_list = obs # Use obs directly as obstacle_list for clarity

    sx = 0.0
    sy = 55.0
    gx = AREA_WIDTH - 5.0
    gy = 5.0
    reso = 1  # potential grid size [m]
    robot_radius = 3.0  # robot radius [m] - reduced robot radius for APF

    # Initialize RRTVisualizer (can be reused or adapted for PF)
    visualizer = RRTVisualizer( # Reusing RRTVisualizer, might need adaptation
        play_area=[0, AREA_WIDTH, 0, AREA_WIDTH], # Adjust play area to AREA_WIDTH
        start=RRTPlanner.Node(sx, sy),  # Using RRTPlanner.Node for consistency
        end=RRTPlanner.Node(gx, gy),
        robot_radius=robot_radius,
        obstacle_list=obstacle_list
    )

    # Initialize PotentialFieldPlanner
    pf_planner = PotentialFieldPlanner(
        obstacle_list=obstacle_list,
        goal=(gx, gy),
        rand_area=[-1, 60], # rand_area might not be directly used but kept for interface
        reso=reso,
        robot_radius=robot_radius,
        play_area=[0, AREA_WIDTH, 0, AREA_WIDTH], # Adjust play area to AREA_WIDTH
        visualizer=visualizer # Pass visualizer
    )


    if show_animation:
        plt.grid(True)
        plt.axis("equal")
        plt.xlim(0, AREA_WIDTH)
        plt.ylim(0, AREA_WIDTH)

    # Path generation using Potential Field Planner
    rx, ry, path_length = pf_planner.dynamic_potential_field_planning(
        sx, sy, gx, gy,
        reso=reso,
        rr=robot_radius,
        enable_heatmap=enable_heatmap,
        show_animation=show_animation
    )

    if show_animation:
        # Add path length information to the figure
        plt.figtext(0.5, 0.82, f"Total path length: {path_length:.2f}m",
                    ha="center", fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))
        plt.show()


if __name__ == '__main__':
    main()