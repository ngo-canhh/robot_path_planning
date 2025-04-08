import matplotlib.pyplot as plt
from shape import Circle, Rectangle 
from obstacle import StaticObstacle, DynamicObstacle, ObstacleDetector 
from ray_tracing import RayTracing 
from waiting_rule import WaitingRule 
from smoother import ShortcutSmoother, BezierSmooth, Pipeline 
from rrt_planner import RRTPlanner, RRTStarPlanner 

show_animation = True 

def main(gx=55.0, gy=5.0, random_seed=None):
    print("start " + __file__)

    # plt.figure(figsize=(12,10))

    # detector = ObstacleDetector()
    # positions = {
    #     "chair.png": (25, 10),
    #     "table.png": (40, 10),
    #     "cat.png": (30, 25), 
    #     "dog.png": (34, 21), 
        
    # }
    # sizes = {
    #     "chair.png": (20, 30),
    #     "table.png": (20, 30),
    #     "cat.png": (12, 12),
    #     "dog.png": (20, 20),
    # }
    # velocities = {
    #     "cat.png": (-1, 0), 
    #     "dog.png": (-1, -1),
    # }
    
    # static_obstacles, dynamic_obstacles = detector.detect_obstacles_from_folder(positions=positions, sizes=sizes, velocities=velocities)

    static_obstacles = [
        # Top-left rectangle (tall)
          # Top-left rectangle (short)
        StaticObstacle(Rectangle(20, 5, 8, 8)),

        StaticObstacle(Rectangle(-2, 35, 5, 10)),

        # Oval/Circle in the middle-left
        StaticObstacle(Circle(15, 30, 4)),

        # Triangle at top-right (approximated with a rectangle)
        StaticObstacle(Rectangle(35, 5, 10, 10)),
        StaticObstacle(Rectangle(8, 47, 6, 10)),

        StaticObstacle(Circle(43, 45, 4)),
        #drawback
        StaticObstacle(Rectangle(50, 10, 10, 2)),
        StaticObstacle(Rectangle(50, 0, 2, 8)),
        StaticObstacle(Rectangle(50, 0, 10, 2)),

    ]
    dynamic_obstacles = [
        DynamicObstacle(Rectangle(27, 25, 16, 2), -1,-1), 
        DynamicObstacle(Rectangle(34, 18, 2, 16), -1,-1) ,
    ]
    obstacle_list = static_obstacles + dynamic_obstacles

    ray_tracer = RayTracing(static_obstacles)
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
        smoother=Pipeline([BezierSmooth()])
    )

    # rrt_planner = RRTStarPlanner(
    #     start=[0, 55],           
    #     goal=[gx, gy],         
    #     rand_area=[-1, 60],
    #     obstacle_list=obstacle_list,
    #     robot_radius=0.8,
    #     play_area=[-5.0, 60.0, -5.0, 60.0],
    #     path_resolution=1,
    #     ray_tracer=ray_tracer,
    #     waiting_rule=waiting_rule,
    #     random_seed=random_seed,
    #     show_animation=True,
    #     smoother=None
    # )
    typeRRT = "RRT*" if (isinstance(rrt_planner, RRTStarPlanner)) else "RRT"

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
            plt.text(30, 57, f"Path Length: {rrt_planner.path_length:.2f}",
                     bbox=dict(facecolor='white', alpha=0.7))
            plt.plot([x for (x, y) in planner_path], [y for (x, y) in planner_path], '-r', label=typeRRT)
            plt.plot([x for (x, y) in smoother_path], [y for (x, y) in smoother_path], '-b', label="Smoothed path")

            plt.grid(True)
            # plt.pause(0.01)
            plt.legend()
            plt.show()


if __name__ == '__main__':
    main(random_seed=10)  