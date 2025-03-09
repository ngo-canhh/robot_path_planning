import matplotlib.pyplot as plt # If you want to plot
from shape import Circle, Rectangle  # Import Shape classes
from obstacle import StaticObstacle, DynamicObstacle # Import Obstacle classes
from ray_tracing import RayTracing # Import RayTracing
from waiting_rule import WaitingRule # Import WaitingRule
from rrt_planner import RRTPlanner # Import RRTPlanner

show_animation = True # Flag to enable/disable animation

def main(gx=6.0, gy=10.0):
    print("start " + __file__)

    # ====Search Path with RRT====
    static_obstacle_list_data = [
        (5, 5, 1), # Circle
        (3, 6, 2), (3, 8, 2), (3, 10, 2), # Circles
        (7, 5, 2), (9, 5, 2), (8, 10, 1), # Circles
        (0, 0, 2, 2) # Rectangle (x, y, width, height)
    ]  # [x, y, radius] or [x, y, width, height]
    dynamic_obstacle_list_data = [
        ((4, 8), (0.5, -0.2), 1), # Dynamic Circle: (position, velocity, radius)
        # Add more dynamic obstacles if needed
    ]

    static_obstacles = []
    for data in static_obstacle_list_data:
        if len(data) == 3: # Circle
            static_obstacles.append(StaticObstacle(shape=Circle(x=data[0], y=data[1], radius=data[2])))
        elif len(data) == 4: # Rectangle
            static_obstacles.append(StaticObstacle(shape=Rectangle(x=data[0], y=data[1], width=data[2], height=data[3])))

    dynamic_obstacles = [
        DynamicObstacle(shape=Circle(x=pos[0], y=pos[1], radius=radius), vx=vel[0], vy=vel[1])
        for pos, vel, radius in dynamic_obstacle_list_data
    ]
    obstacle_list = static_obstacles + dynamic_obstacles # Combined obstacle list

    # Initialize RayTracing and WaitingRule
    ray_tracer = RayTracing(static_obstacles) # RayTracing for static obstacles
    waiting_rule = WaitingRule(robot_speed=1.0) # WaitingRule for dynamic obstacles

    # Set Initial parameters
    rrt_planner = RRTPlanner( # Use RRTPlanner instead of original RRT
        start=[0, 0],
        goal=[gx, gy],
        rand_area=[-2, 15],
        obstacle_list=obstacle_list,
        robot_radius=0.8,
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