# --- START OF FILE visualize_trace_rays.py ---

import numpy as np
import matplotlib.pyplot as plt
import math

# Assuming the files are structured as described above
from utils.ray_tracing_algorithm import RayTracingAlgorithm
from components.obstacle import StaticObstacle, DynamicObstacle, Obstacle
from components.shape import Circle, Rectangle, Triangle, Polygon, Shape # Import specific shapes

# --- Helper function to plot robot ---
def plot_robot(ax, x, y, radius, orientation):
    # Robot body
    robot_circle = plt.Circle((x, y), radius, color='blue', fill=False, linewidth=1.5, zorder=5)
    ax.add_patch(robot_circle)
    # Orientation line
    arrow_len = radius * 1.5
    ax.arrow(x, y,
             arrow_len * math.cos(orientation),
             arrow_len * math.sin(orientation),
             head_width=radius*0.5, head_length=radius*0.5, fc='blue', ec='blue', zorder=6)
    ax.plot(x, y, 'bo', markersize=3) # Center point

# --- Helper function to plot obstacles ---
def plot_obstacles(ax, obstacles: list):
    for obs in obstacles:
        patch = obs.get_render_patch(alpha=0.7) # Use obstacle's render method
        ax.add_patch(patch)
        # Optionally plot obstacle center
        # ax.plot(obs.x, obs.y, 'kx', markersize=4)

# --- Main Visualization Logic ---
def visualize_trace_rays():
    # 1. Environment Setup
    ENV_WIDTH = 200
    ENV_HEIGHT = 150
    ROBOT_RADIUS = 5.0

    # 2. Robot Setup
    robot_x = 30
    robot_y = 40
    robot_orientation = np.pi / 4 # 45 degrees

    # 3. Obstacle Setup (Create a mix of shapes)
    obstacles = [
        StaticObstacle(x=130, y=75, shape=Circle(radius=15)),
        StaticObstacle(x=95, y=100, shape=Rectangle(width=30, height=20, angle=np.pi/6)),
        StaticObstacle(x=40, y=100, shape=Triangle(vertices=[(-10,-8), (12, -8), (0, 15)])),
        StaticObstacle(x=150, y=30, shape=Polygon(vertices=[(0,0),(20,5),(15,25),(-5,20)])),
        # Add a dynamic one for visual difference if needed, though trace_rays treats all as static geometry
        # DynamicObstacle(x=160, y=120, shape=Circle(radius=8), velocity=10, direction=np.array([-1,-0.5]))
    ]

    # 4. Initialize RayTracer
    ray_tracer = RayTracingAlgorithm(ENV_WIDTH, ENV_HEIGHT, ROBOT_RADIUS)

    # 5. Call the function to visualize
    NUM_RAYS = 48  # Increase for finer detail
    MAX_RAY_LENGTH = 180
    ray_intersections, ray_viz_points = ray_tracer.trace_rays(
        robot_x, robot_y, robot_orientation, obstacles,
        num_rays=NUM_RAYS, max_ray_length=MAX_RAY_LENGTH
    )

    # 6. Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f'Visualization of trace_rays ({NUM_RAYS} rays)')
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_xlim(0, ENV_WIDTH)
    ax.set_ylim(0, ENV_HEIGHT)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Plot environment boundaries (optional, good for context)
    ax.plot([0, ENV_WIDTH, ENV_WIDTH, 0, 0], [0, 0, ENV_HEIGHT, ENV_HEIGHT, 0], 'k-', label='Environment Boundary')

    # Plot Robot
    plot_robot(ax, robot_x, robot_y, ROBOT_RADIUS, robot_orientation)

    # Plot Obstacles
    plot_obstacles(ax, obstacles)

    # Plot Rays and Intersections
    for i, viz_data in enumerate(ray_viz_points):
        start_point, end_point = viz_data
        intersect_x, intersect_y, hit_object = ray_intersections[i]

        # Draw the ray line
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                color='cyan', linestyle='-', linewidth=0.5, alpha=0.8, zorder=2)

        # Mark the intersection point
        if hit_object: # If it hit something before max length
            color = 'red' if isinstance(hit_object, Obstacle) else 'orange' if hit_object == "Boundary" else 'gray' # Should not be gray
            marker = 'x' if isinstance(hit_object, Obstacle) else '*' if hit_object == "Boundary" else '.'
            size = 6 if isinstance(hit_object, Obstacle) else 8 if hit_object == "Boundary" else 4
            ax.plot(intersect_x, intersect_y, marker=marker, color=color, markersize=size, zorder=3, label=f'{type(hit_object).__name__} Hit' if i==0 and isinstance(hit_object, Obstacle) else '_nolegend_')
            ax.plot(intersect_x, intersect_y, marker=marker, color=color, markersize=size, zorder=3, label='Boundary Hit' if i==0 and hit_object == "Boundary" else '_nolegend_')

        else: # Ray reached max length without hitting anything specific (or hit boundary at max length)
             ax.plot(intersect_x, intersect_y, marker='.', color='gray', markersize=3, zorder=3, label='Max Length' if i == 0 else '_nolegend_')


    # Create legend with unique labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Remove duplicate labels
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.show()

# --- Run the visualization ---
if __name__ == "__main__":
    visualize_trace_rays()

# --- END OF FILE visualize_trace_rays.py ---