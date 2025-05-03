# --- START OF FILE visualize_avoid_static_obstacle.py ---

import numpy as np
import matplotlib.pyplot as plt
import math

# Assuming the files are structured as described above
from utils.ray_tracing_algorithm import RayTracingAlgorithm
from components.obstacle import StaticObstacle, Obstacle
from components.shape import Circle, Rectangle, Triangle, Polygon # Import specific shapes

# --- Helper functions (can reuse from visualize_trace_rays.py or copy here) ---
def plot_robot(ax, x, y, radius, orientation):
    robot_circle = plt.Circle((x, y), radius, color='blue', fill=False, linewidth=1.5, zorder=5)
    ax.add_patch(robot_circle)
    arrow_len = radius * 1.5
    ax.arrow(x, y,
             arrow_len * math.cos(orientation),
             arrow_len * math.sin(orientation),
             head_width=radius*0.5, head_length=radius*0.5, fc='blue', ec='blue', zorder=6)
    ax.plot(x, y, 'bo', markersize=3)

def plot_obstacles(ax, obstacles: list):
    for obs in obstacles:
        patch = obs.get_render_patch(alpha=0.7)
        ax.add_patch(patch)

# --- Main Visualization Logic ---
def visualize_avoid_obstacle():
    # 1. Environment Setup
    ENV_WIDTH = 200
    ENV_HEIGHT = 150
    ROBOT_RADIUS = 5.0

    # 2. Robot Setup
    robot_x = 143
    robot_y = 74
    robot_orientation = np.pi * -1/6 # Initial orientation

    # 3. Obstacle Setup (Focus on ONE obstacle for this function)
    # obstacle_to_avoid = StaticObstacle(x=90, y=75, shape=Circle(radius=20))
    # obstacle_to_avoid = StaticObstacle(x=100, y=70, shape=Rectangle(width=40, height=25, angle=-np.pi/8))
    # obstacle_to_avoid = StaticObstacle(x=80, y=90, shape=Triangle(vertices=[(-15,-10), (15, -10), (0, 20)]))
    obstacle_to_avoid = StaticObstacle(x=90, y=70, shape=Polygon(vertices=[(-10,-10),(30,-15),(50,10),(0,20),(-15,5)]))


    # Other obstacles for context (optional, trace_rays uses them, avoid_obstacle ignores them)
    other_obstacles = [
         StaticObstacle(x=150, y=120, shape=Circle(radius=10)),
         StaticObstacle(x=30, y=110, shape=Rectangle(width=15, height=40, angle=0)),
        #  StaticObstacle(x=50, y=30, shape=Polygon(vertices=[(-10,-8), (12, -8), (0, 15), (5, 5), (-5, 5), (7, 35)])),
    ]
    all_obstacles = [obstacle_to_avoid] + other_obstacles


    # 4. Lookahead Point (Target direction)
    lookahead_x = 96
    lookahead_y = 104
    # lookahead_x = 75 # Test case: Goal behind obstacle
    # lookahead_y = 75

    # 5. Initialize RayTracer (using its parameters for internal ray casting)
    ray_tracer = RayTracingAlgorithm(ENV_WIDTH, ENV_HEIGHT, ROBOT_RADIUS)
    # Get parameters needed for visualization ray casting
    num_rays = ray_tracer.num_avoidance_rays
    angle_span = ray_tracer.avoidance_ray_angle_span
    max_dist = ray_tracer.avoidance_ray_max_dist
    safety_ratio = ray_tracer.safety_distance_ratio

    # --- 6. Replicate Internal Ray Casting for Visualization ---
    # (This part mimics the logic inside avoid_static_obstacle to get ray data)
    robot_pos = np.array([robot_x, robot_y])
    vec_to_goal = np.array([lookahead_x - robot_x, lookahead_y - robot_y])
    dist_to_goal = np.linalg.norm(vec_to_goal)
    if dist_to_goal < 1e-6:
        angle_to_goal = robot_orientation
    else:
        angle_to_goal = np.arctan2(vec_to_goal[1], vec_to_goal[0])

    ray_angles = np.linspace(
        angle_to_goal - angle_span / 2,
        angle_to_goal + angle_span / 2,
        num_rays
    )

    viz_rays = [] # Store tuples: (start_pos, end_pos, is_clear)

    if not (hasattr(obstacle_to_avoid.shape, 'intersect_ray') and callable(obstacle_to_avoid.shape.intersect_ray)):
         print(f"Warning: Obstacle shape {type(obstacle_to_avoid.shape)} lacks intersect_ray. Cannot visualize rays.")
         # Handle error or skip ray plotting
    else:
        obs_x, obs_y = obstacle_to_avoid.x, obstacle_to_avoid.y
        for angle in ray_angles:
            norm_angle = ray_tracer._normalize_angle(angle)
            ray_dir = np.array([np.cos(norm_angle), np.sin(norm_angle)])

            # Check intersection with the *specific obstacle*
            intersect_t = obstacle_to_avoid.shape.intersect_ray(robot_pos, ray_dir, obs_x, obs_y)
            safe_intersect_t = max(0, intersect_t - ROBOT_RADIUS * safety_ratio)

            # Check boundaries
            boundary_t = float('inf')
            if abs(ray_dir[0]) > 1e-6 and ray_dir[0] < 0: t_b = -robot_x / ray_dir[0]; boundary_t = min(boundary_t, t_b)
            if abs(ray_dir[0]) > 1e-6 and ray_dir[0] > 0: t_b = (ENV_WIDTH - robot_x) / ray_dir[0]; boundary_t = min(boundary_t, t_b)
            if abs(ray_dir[1]) > 1e-6 and ray_dir[1] < 0: t_b = -robot_y / ray_dir[1]; boundary_t = min(boundary_t, t_b)
            if abs(ray_dir[1]) > 1e-6 and ray_dir[1] > 0: t_b = (ENV_HEIGHT - robot_y) / ray_dir[1]; boundary_t = min(boundary_t, t_b)
            # Ensure boundary check respects non-negative t
            boundary_t = max(0, boundary_t)


            final_t = min(safe_intersect_t, boundary_t, max_dist)
            end_point = tuple(robot_pos + ray_dir * final_t)
            is_clear = final_t >= max_dist * 0.98 # Check if ray reached close to max distance

            viz_rays.append(((robot_x, robot_y), end_point, is_clear))

    # 7. Call the actual function to get the chosen angle
    # Check for emergency case first (optional but good for complete viz)
    dist_to_boundary = obstacle_to_avoid.get_efficient_distance(robot_x, robot_y)
    if dist_to_boundary <= ROBOT_RADIUS:
         print("Visualization: Robot is potentially overlapping (Emergency case)")
         # Plot vector away?
         vec_away = obstacle_to_avoid.shape.get_effective_vector(robot_x, robot_y, obstacle_to_avoid.x, obstacle_to_avoid.y)
         if np.linalg.norm(vec_away) > 1e-6:
             chosen_avoidance_angle = np.arctan2(vec_away[1], vec_away[0])
             emergency_case = True
         else: # Deep overlap or error
             chosen_avoidance_angle = ray_tracer._normalize_angle(robot_orientation + np.pi) # Turn around
             emergency_case = True
         print(f"Emergency avoidance angle: {math.degrees(chosen_avoidance_angle):.1f} deg")
    else:
        chosen_avoidance_angle = ray_tracer.avoid_static_obstacle(
            robot_x, robot_y, ROBOT_RADIUS, robot_orientation,
            obstacle_to_avoid, lookahead_x, lookahead_y
        )
        emergency_case = False
        print(f"Calculated avoidance angle: {math.degrees(chosen_avoidance_angle):.1f} deg")


    # 8. Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    title = 'Visualization of avoid_static_obstacle'
    if emergency_case: title += ' (Emergency Triggered)'
    ax.set_title(title)

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_xlim(0, ENV_WIDTH)
    ax.set_ylim(0, ENV_HEIGHT)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Plot environment boundaries
    ax.plot([0, ENV_WIDTH, ENV_WIDTH, 0, 0], [0, 0, ENV_HEIGHT, ENV_HEIGHT, 0], 'k-', label='Environment Boundary')

    # Plot Robot
    plot_robot(ax, robot_x, robot_y, ROBOT_RADIUS, robot_orientation)

    # Plot Obstacles (highlight the one being avoided)
    plot_obstacles(ax, other_obstacles) # Plot others first
    avoid_patch = obstacle_to_avoid.get_render_patch(alpha=0.8, zorder=4) # Highlighted
    ax.add_patch(avoid_patch)
    ax.plot([], [], color='red', alpha=0.8, label='Obstacle to Avoid') # Legend entry

    # Plot Lookahead Point and Desired Direction
    ax.plot(lookahead_x, lookahead_y, 'go', markersize=8, label='Lookahead Point')
    ax.plot([robot_x, lookahead_x], [robot_y, lookahead_y], 'g:', linewidth=1.5, label='Direction to Lookahead')

    # Plot the avoidance rays cast for visualization
    plotted_clear = False
    plotted_blocked = False
    for start, end, is_clear in viz_rays:
        color = 'orange' if is_clear else 'darkgoldenrod'
        linestyle = '-' if is_clear else '--'
        linewidth = 1.0 if is_clear else 0.8
        alpha = 0.9 if is_clear else 0.7
        label = None
        if is_clear and not plotted_clear:
            label = 'Clear Ray (Avoidance Scan)'
            plotted_clear = True
        elif not is_clear and not plotted_blocked:
            label = 'Blocked Ray (Avoidance Scan)'
            plotted_blocked = True

        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linestyle=linestyle,
                linewidth=linewidth, alpha=alpha, zorder=2, label=label)
        if not is_clear:
             ax.plot(end[0], end[1], 'x', color='darkred', markersize=4, zorder=3) # Mark blocked endpoint


    # Plot the chosen avoidance direction
    avoid_arrow_len = ROBOT_RADIUS * 4
    ax.arrow(robot_x, robot_y,
             avoid_arrow_len * math.cos(chosen_avoidance_angle),
             avoid_arrow_len * math.sin(chosen_avoidance_angle),
             head_width=ROBOT_RADIUS*0.6, head_length=ROBOT_RADIUS*0.7,
             fc='magenta', ec='magenta', linewidth=2, zorder=7,
             label='Chosen Avoidance Direction')


    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=8)

    plt.show()


# --- Run the visualization ---
if __name__ == "__main__":
    visualize_avoid_obstacle()

# --- END OF FILE visualize_avoid_static_obstacle.py ---