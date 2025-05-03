# --- START OF FILE visualize_shape_methods.py ---
import matplotlib.pyplot as plt
import numpy as np
import math

# Assuming components folder is accessible
from components.shape import Shape, Circle, Rectangle, Triangle, Polygon

# --- Helper Functions ---
def plot_robot(ax, x, y, radius, color='blue', fill=False, label='Robot'):
    """Plots the robot as a circle."""
    robot_circle = plt.Circle((x, y), radius, color=color, fill=fill, zorder=5, label=label)
    ax.add_patch(robot_circle)

def setup_plot(title, xlim=(0, 100), ylim=(0, 100)):
    """Sets up a standard Matplotlib plot."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    return fig, ax

# --- Visualization Functions for Shape Methods ---

def visualize_check_collision(shape: Shape, obs_x, obs_y, robot_x, robot_y, robot_radius):
    """Visualizes shape.check_collision."""
    collides = shape.check_collision(robot_x, robot_y, robot_radius, obs_x, obs_y)
    robot_color = 'red' if collides else 'green'
    title = f'check_collision ({shape.__class__.__name__}) - Result: {collides}'

    fig, ax = setup_plot(title, xlim=(obs_x - 100, obs_x + 100), ylim=(obs_y - 100, obs_y + 100))

    # Plot Shape
    shape_patch = shape.get_patch(obs_x, obs_y, color='gray', alpha=0.7, zorder=3)
    ax.add_patch(shape_patch)

    # Plot Robot
    plot_robot(ax, robot_x, robot_y, robot_radius, color=robot_color, fill=True, label=f'Robot (Collision: {collides})')

    ax.legend()
    plt.show()

def visualize_intersects_segment(shape: Shape, obs_x, obs_y, p1, p2, robot_radius):
    """Visualizes shape.intersects_segment."""
    intersects = shape.intersects_segment(p1, p2, robot_radius, obs_x, obs_y)
    segment_color = 'red' if intersects else 'green'
    title = f'intersects_segment ({shape.__class__.__name__}) - Result: {intersects}'

    # Adjust plot limits based on shape, segment, and obstacle position
    all_x = [p1[0], p2[0], obs_x - shape.get_effective_radius(), obs_x + shape.get_effective_radius()]
    all_y = [p1[1], p2[1], obs_y - shape.get_effective_radius(), obs_y + shape.get_effective_radius()]
    xlim = (min(all_x) - 20, max(all_x) + 20)
    ylim = (min(all_y) - 20, max(all_y) + 20)

    fig, ax = setup_plot(title, xlim=xlim, ylim=ylim)

    # Plot Shape
    shape_patch = shape.get_patch(obs_x, obs_y, color='gray', alpha=0.7, zorder=3)
    ax.add_patch(shape_patch)

    # Plot Segment
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=segment_color, marker='o', linestyle='-', linewidth=2,
            label=f'Segment (Intersects: {intersects}) - Radius Padding: {robot_radius}')

    # Optional: Visualize the radius padding around the segment (crude approximation)
    # For a better visualization, one might draw circles at endpoints or sample points
    ax.plot(p1[0], p1[1], 'o', markersize=robot_radius*0.5, color=segment_color, alpha=0.3) # Rough indication
    ax.plot(p2[0], p2[1], 'o', markersize=robot_radius*0.5, color=segment_color, alpha=0.3) # Rough indication


    ax.legend()
    plt.show()

def visualize_get_efficient_distance(shape: Shape, obs_x, obs_y, point_x, point_y):
    """Visualizes shape.get_efficient_distance."""
    distance = shape.get_efficient_distance(point_x, point_y, obs_x, obs_y)
    title = f'get_efficient_distance ({shape.__class__.__name__})'

    fig, ax = setup_plot(title, xlim=(obs_x - 100, obs_x + 100), ylim=(obs_y - 100, obs_y + 100))

    # Plot Shape
    shape_patch = shape.get_patch(obs_x, obs_y, color='gray', alpha=0.7, zorder=3)
    ax.add_patch(shape_patch)

    # Plot Point
    ax.plot(point_x, point_y, 'bo', markersize=8, label='Test Point')

    # Display Distance
    ax.text(point_x + 5, point_y + 5, f'Dist: {distance:.2f}', color='blue', fontsize=12, zorder=6)

    # Optional: Draw line to closest point (requires get_effective_vector)
    try:
        vec = shape.get_effective_vector(point_x, point_y, obs_x, obs_y)
        closest_pt = (point_x + vec[0], point_y + vec[1])
        ax.plot([point_x, closest_pt[0]], [point_y, closest_pt[1]], 'b--', linewidth=1, label='Line to Boundary')
        ax.plot(closest_pt[0], closest_pt[1], 'bx', markersize=6)
    except Exception as e:
        print(f"Could not visualize distance line: {e}")


    ax.legend()
    plt.show()

def visualize_get_effective_vector(shape: Shape, obs_x, obs_y, point_x, point_y):
    """Visualizes shape.get_effective_vector."""
    vector = shape.get_effective_vector(point_x, point_y, obs_x, obs_y)
    closest_point = (point_x + vector[0], point_y + vector[1])
    title = f'get_effective_vector ({shape.__class__.__name__})'

    fig, ax = setup_plot(title, xlim=(obs_x - 100, obs_x + 100), ylim=(obs_y - 100, obs_y + 100))

    # Plot Shape
    shape_patch = shape.get_patch(obs_x, obs_y, color='gray', alpha=0.7, zorder=3)
    ax.add_patch(shape_patch)

    # Plot Point
    ax.plot(point_x, point_y, 'ro', markersize=8, label='Test Point (P)')

    # Plot Closest Point on Boundary
    ax.plot(closest_point[0], closest_point[1], 'rx', markersize=8, label='Closest Point (C)')

    # Plot Vector (from P to C)
    ax.arrow(point_x, point_y, vector[0], vector[1],
             head_width=5, head_length=7, fc='red', ec='red', linestyle='-', zorder=7,
             label=f'Vector P->C: ({vector[0]:.1f}, {vector[1]:.1f})')

    ax.legend()
    plt.show()


if __name__ == "__main__":
    # --- Define Shapes ---
    circle = Circle(radius=30)
    rectangle = Rectangle(width=60, height=40, angle=np.pi / 6)
    triangle = Triangle([(-20, 25), (40, 0), (-10, -35)])
    polygon = Polygon([(0,0), (50,10), (40,60), (-20, 50), (-30, 15)])
    shapes = [circle, rectangle, triangle, polygon]
    obs_x, obs_y = 50, 50 # Center position for the shapes

    # --- Test Parameters ---
    robot_radius = 10
    # Collision Test Points
    robot_pos_collide = (60, 55) # Likely collide with circle/rect
    robot_pos_no_collide = (10, 10) # Likely no collision
    # Segment Test Points
    seg_p1_intersect = (0, 70)
    seg_p2_intersect = (100, 30) # Likely intersects some shapes
    seg_p1_no_intersect = (0, 0)
    seg_p2_no_intersect = (20, 20) # Likely doesn't intersect
    # Distance/Vector Test Points
    point_inside = (55, 55) # Inside most shapes centered at 50,50
    point_outside = (100, 100) # Outside all shapes
    point_edge = (80, 50) # Near edge of circle


    # --- Run Visualizations ---
    for shape in shapes:
        print(f"\n--- Visualizing: {shape.__class__.__name__} ---")

        # 1. check_collision
        visualize_check_collision(shape, obs_x, obs_y, robot_pos_collide[0], robot_pos_collide[1], robot_radius)
        visualize_check_collision(shape, obs_x, obs_y, robot_pos_no_collide[0], robot_pos_no_collide[1], robot_radius)

        # 2. intersects_segment
        visualize_intersects_segment(shape, obs_x, obs_y, seg_p1_intersect, seg_p2_intersect, robot_radius)
        visualize_intersects_segment(shape, obs_x, obs_y, seg_p1_no_intersect, seg_p2_no_intersect, robot_radius)

        # 3. get_efficient_distance
        visualize_get_efficient_distance(shape, obs_x, obs_y, point_inside[0], point_inside[1])
        visualize_get_efficient_distance(shape, obs_x, obs_y, point_outside[0], point_outside[1])
        visualize_get_efficient_distance(shape, obs_x, obs_y, point_edge[0], point_edge[1]) # Near circle edge

        # 4. get_effective_vector
        visualize_get_effective_vector(shape, obs_x, obs_y, point_inside[0], point_inside[1])
        visualize_get_effective_vector(shape, obs_x, obs_y, point_outside[0], point_outside[1])
        visualize_get_effective_vector(shape, obs_x, obs_y, point_edge[0], point_edge[1]) # Near circle edge

# --- END OF FILE visualize_shape_methods.py ---