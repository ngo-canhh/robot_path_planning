import matplotlib.pyplot as plt
from shape import Circle, Rectangle  # Import Shape classes
from obstacle import StaticObstacle, DynamicObstacle, ObstacleDetector # Import Obstacle classes
from ray_tracing import RayTracing # Import RayTracing
from waiting_rule import WaitingRule # Import WaitingRule
from smoother import ShortcutSmoother, BezierSmooth, Pipeline # Import ShortcutSmooth
from rrt_planner import RRTPlanner, RRTVisualizer # Import RRTPlanner and RRTVisualizer

show_animation = True # Flag to enable/disable animation

def main(gx=55.0, gy=5.0, random_seed=None):
    print("start " + __file__)

    plt.figure(figsize=(12,10))

    detector = ObstacleDetector()
    positions = {
        "chair.png": (30, 10),
        "table.png": (40, 10),
        "cat.png": (30, 25), 
        "dog.png": (34, 21), 
        
    }
    sizes = {
        "chair.png": (20, 15),
        "table.png": (15, 15),
        "cat.png": (12, 12),
        "dog.png": (20, 20),
    }
    velocities = {
        "cat.png": (-1, 0), 
        "dog.png": (-1, -1),
    }
    
    static_obstacles, dynamic_obstacles = detector.detect_obstacles_from_folder(positions=positions, sizes=sizes, velocities=velocities)

    no_image_obstacles = [
        StaticObstacle(Circle(10, 10, 5)),
        StaticObstacle(Rectangle(45, 35, 10, 10)),
        StaticObstacle(Rectangle(10, 30, 20, 20)),
        StaticObstacle(Rectangle(10, 10, 40, 5)),
    ]
    obstacle_list = static_obstacles + dynamic_obstacles + no_image_obstacles

    # Initialize RayTracing and WaitingRule
    ray_tracer = RayTracing(static_obstacles + no_image_obstacles)
    waiting_rule = WaitingRule(robot_speed=1.0)
    
    rrt_planner = RRTPlanner(
        start=[0, 55],           
        goal=[gx, gy],         
        rand_area=[-1, 60],
        obstacle_list=obstacle_list,
        robot_radius=0.8,
        play_area=[-5.0, 60.0, -5.0, 60.0],
        path_resolution=1,
        ray_tracer=ray_tracer,
        waiting_rule=waiting_rule,
        max_iter=1000,
        random_seed=random_seed,
        show_animation = True,
        smoother=Pipeline([ShortcutSmoother(ray_tracer), ])
    )
    rrt_planner.planning(animation=show_animation)
    planner_path = rrt_planner.planner_path

    if planner_path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        smoother_path = rrt_planner.final_path

        # Draw final path
        if show_animation and rrt_planner.visualizer: 
            rrt_planner.visualizer.draw_graph(rnd_node=None, rrt_planner=rrt_planner) 
            # Add path length text to the figure
            plt.text(30, 57, f"Path Length: {rrt_planner.path_length:.2f}",
                     bbox=dict(facecolor='white', alpha=0.7))
            plt.plot([x for (x, y) in planner_path], [y for (x, y) in planner_path], '-r', label="RRT path")
            plt.plot([x for (x, y) in smoother_path], [y for (x, y) in smoother_path], '-b', label="Smoothed path")

            plt.grid(True)
            # plt.pause(0.01)
            plt.legend()
            plt.show()


if __name__ == '__main__':
    main(random_seed=10)  