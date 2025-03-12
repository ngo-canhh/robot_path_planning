import matplotlib.pyplot as plt # If you want to plot
from shape import Circle, Rectangle  # Import Shape classes
from obstacle import StaticObstacle, DynamicObstacle # Import Obstacle classes
from ray_tracing import RayTracing # Import RayTracing
from waiting_rule import WaitingRule # Import WaitingRule
from rrt_planner import RRTPlanner # Import RRTPlanner

show_animation = True # Flag to enable/disable animation

def main(gx=55.0, gy=55.0):
    print("start " + __file__)

    static_obstacles = [
        # Vật cản tĩnh hình tròn
      StaticObstacle(Circle(10, 10, 3)),
      StaticObstacle(Circle(10, 25, 4)),
      StaticObstacle(Circle(30, 15, 3.5)),
      StaticObstacle(Circle(45, 30, 5)),
      StaticObstacle(Circle(50, 50, 2.5)),

      # Vật cản tĩnh hình chữ nhật
      StaticObstacle(Rectangle(20, 5, 5, 3)),
      StaticObstacle(Rectangle(35, 20, 4, 6)),
      StaticObstacle(Rectangle(5, 40, 6, 2)),
      StaticObstacle(Rectangle(40, 45, 3, 7)),
    ]
    dynamic_obstacles = [
      # Vật cản động hình tròn
      DynamicObstacle(Circle(15, 35, 2), 1, 0),      # Di chuyển ngang
      DynamicObstacle(Circle(25, 45, 2.5), 0, -0.5),   # Di chuyển dọc
      DynamicObstacle(Circle(55, 10, 3), -0.8, 0.8),  # Di chuyển chéo

      # Vật cản động hình chữ nhật
      DynamicObstacle(Rectangle(35, 35, 2, 4), -0.5, 0), # Di chuyển ngang
      DynamicObstacle(Rectangle(45, 10, 3, 2), 0, 0.7),  # Di chuyển dọc
      DynamicObstacle(Rectangle(5, 5, 4, 3), 0.6, -0.6)   # Di chuyển chéo
    ]

    # static_obstacles = []
    # for data in static_obstacle_list_data:
    #     if len(data) == 3: # Circle
    #         static_obstacles.append(StaticObstacle(shape=Circle(x=data[0], y=data[1], radius=data[2])))
    #     elif len(data) == 4: # Rectangle
    #         static_obstacles.append(StaticObstacle(shape=Rectangle(x=data[0], y=data[1], width=data[2], height=data[3])))

    # dynamic_obstacles = [
    #     DynamicObstacle(shape=Circle(x=pos[0], y=pos[1], radius=radius), vx=vel[0], vy=vel[1])
    #     for pos, vel, radius in dynamic_obstacle_list_data
    # ]
    obstacle_list = static_obstacles + dynamic_obstacles # Combined obstacle list

    # Initialize RayTracing and WaitingRule
    ray_tracer = RayTracing(static_obstacles) # RayTracing for static obstacles
    waiting_rule = WaitingRule(robot_speed=1.0) # WaitingRule for dynamic obstacles

    # Set Initial parameters
    rrt_planner = RRTPlanner( # Use RRTPlanner instead of original RRT
        start=[0, 0],
        goal=[gx, gy],
        rand_area=[-1, 60],
        obstacle_list=obstacle_list,
        robot_radius=0.8,
        play_area=[-5.0, 60.0, -5.0, 60.0],
        ray_tracer=ray_tracer, # Pass RayTracing object
        waiting_rule=waiting_rule # Pass WaitingRule object
    )
    path = rrt_planner.planning(animation = show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            rrt_planner.draw_graph() # Need to adjust RRTPlanner.draw_graph to draw correctly
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()


if __name__ == '__main__':
    main()