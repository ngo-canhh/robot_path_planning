import matplotlib.pyplot as plt
from shape import Circle, Rectangle  
from obstacle import StaticObstacle, DynamicObstacle 
from ray_tracing import RayTracing 
from waiting_rule import WaitingRule 
from rrt_planner import RRTPlanner, RRTVisualizer 

show_animation = True 

def main(gx=20.0, gy=45.0, random_seed=None): 
    print("start " + __file__)

    original_e_rects = [
        Rectangle(9, 10, 5, 40),
        Rectangle(9, 50, 20, 5),
        Rectangle(9, 30, 20, 5),
        Rectangle(9, 10, 20, 5),
    ]

    mirror_e_rects = [
        Rectangle(46, 10, 5, 40),
        Rectangle(31, 50, 20, 5),
        Rectangle(31, 30, 20, 5),
        Rectangle(31, 10, 20, 5),
    ]

    translated_e_obstacles = []
    for rect in original_e_rects:
        translated_rect = Rectangle(rect.x, rect.y, rect.width, rect.height)
        translated_e_obstacles.append(StaticObstacle(translated_rect))

    mirrored_e_obstacles = []
    for rect in mirror_e_rects:
        mirrored_rect = Rectangle(rect.x, rect.y, rect.width, rect.height)
        mirrored_e_obstacles.append(StaticObstacle(mirrored_rect))


    static_obstacles = translated_e_obstacles + mirrored_e_obstacles
    dynamic_obstacles = [] 

    obstacle_list = static_obstacles + dynamic_obstacles;

    ray_tracer = RayTracing(static_obstacles)
    waiting_rule = WaitingRule(robot_speed=1.0)


    rrt_planner = RRTPlanner(
        start=[5, 5],           
        goal=[gx, gy],          
        rand_area=[-1, 60],
        obstacle_list=obstacle_list,
        robot_radius=0.8,
        play_area=[-5.0, 60.0, -5.0, 60.0],
        path_resolution=1,
        ray_tracer=ray_tracer,
        waiting_rule=waiting_rule,
        random_seed=random_seed,
        show_animation=True 
    )
    path = rrt_planner.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        if show_animation: 
            rrt_planner.visualizer.draw_graph(rnd_node=None, rrt_planner=rrt_planner) 
            plt.text(30, 57, f"Path Length: {rrt_planner.path_length:.2f}",
                     bbox=dict(facecolor='white', alpha=0.7))
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)
            plt.show()


if __name__ == '__main__':
    main(random_seed=10)  