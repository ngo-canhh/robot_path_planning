from Obstacle import Obstacle, StaticObstacle, DynamicObstacle
from Shape import Shape, Circle, Rectangle
import numpy as np
from collections import deque
from typing import List
import matplotlib.pyplot as plt

# Parameters
# 
KP = 5.0  # attractive potential gain
ETA = 500.0  # repulsive potential gain
AREA_WIDTH = 30.0  # potential area width [m]
# the number of previous positions used to check oscillations
OSCILLATIONS_DETECTION_LENGTH = 10
show_animation = True

def calc_attractive_potential(x, y, gx, gy):
  return 0.5 * KP * np.hypot(x - gx, y - gy)

# Only nearest obstacle is considered
def calc_repulsive_potential(x, y, obs: List[Obstacle], rr):
  # search nearest obstacle
  min_dist = float("inf")
  for ob in obs:
    d = ob.get_min_distance(x, y)
    if d <= 0:
      return float("inf")
    if d <= min_dist:
      min_dist = d
  if min_dist == float("inf"):
    # print("Error: min_dist is inf")
    return 0.0
  elif min_dist > rr:
    # print("Error: min_dist is greater than rr")
    return 0.0
  else:
    return 0.5 * ETA * (1.0 / min_dist - 1.0 / rr) ** 2

# # All obstacles are considered
# def calc_repulsive_potential(x, y, obs: List[Obstacle], rr):
#   uo = 0.0
#   for ob in obs:
#     d = ob.get_min_distance(x, y)
#     if d <= 0:
#       return float("inf")
#     if d <= rr:
#       uo += 0.5 * ETA * (1.0 / d - 1.0 / rr) ** 2
#   return uo
  
def calc_potential_field(gx, gy, obs: List[Obstacle], reso, rr, sx, sy):
  ox = [ob.get_centroid()[0] for ob in obs]
  oy = [ob.get_centroid()[1] for ob in obs]
        
  minx = min(min(ox), sx, gx) - AREA_WIDTH / 2.0
  miny = min(min(oy), sy, gy) - AREA_WIDTH / 2.0
  maxx = max(max(ox), sx, gx) + AREA_WIDTH / 2.0
  maxy = max(max(oy), sy, gy) + AREA_WIDTH / 2.0
  xw = int(round((maxx - minx) / reso))
  yw = int(round((maxy - miny) / reso))

  # calc each potential
  pmap = [[0.0 for i in range(yw)] for i in range(xw)]

  for ix in range(xw):
      x = ix * reso + minx

      for iy in range(yw):
          y = iy * reso + miny
          ug = calc_attractive_potential(x, y, gx, gy)
          uo = calc_repulsive_potential(x, y, obs, rr)
          if uo == float("inf"):
            pmap[ix][iy] = float("inf")
          else:
            uf = ug + uo
            pmap[ix][iy] = uf

  return pmap, minx, miny, maxx, maxy

def draw_heatmap(data, minx=None, miny=None, maxx=None, maxy=None):
    """
    Draw potential field heatmap aligned with world coordinates
    """
    data = np.array(data).T
    
    if minx is not None and miny is not None and maxx is not None and maxy is not None:
        # Set extent to align heatmap with world coordinates
        extent = [minx, maxx, miny, maxy]
        plt.imshow(data, origin='lower', extent=extent, vmax=100.0, cmap=plt.cm.Blues, interpolation='bilinear')
    else:
        # Fall back to old behavior if coordinates not provided
        plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)

def oscillations_detection(previous_ids, ix, iy):
    previous_ids.append((ix, iy))

    if (len(previous_ids) > OSCILLATIONS_DETECTION_LENGTH):
        previous_ids.popleft()

    # check if contains any duplicates by copying into a set
    previous_ids_set = set()
    for index in previous_ids:
        if index in previous_ids_set:
            return True
        else:
            previous_ids_set.add(index)
    return False

def get_motion_model():
  # dx, dy
  motion = [[1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1]]

  return motion

def dynamic_potential_field_planning(sx, sy, gx, gy, obs: List[Obstacle], reso, rr, dt=0.1, max_time=300.0):
    """
    Path planning with dynamic obstacles using potential field
    """
    # Initial setup similar to original function
    x, y = sx, sy
    rx, ry = [sx], [sy]
    
    # Calculate initial potential field
    ox = [obs.get_centroid()[0] for obs in obs]
    oy = [obs.get_centroid()[1] for obs in obs]
    pmap, minx, miny, maxx, maxy = calc_potential_field(gx, gy, obs, reso, rr, x, y)
    
    # Set up motion model
    motion = get_motion_model()
    previous_ids = deque()
    
    # Simulation time
    time = 0.0
    
    # Distance to goal
    d = np.hypot(x - gx, y - gy)
    ix = round((sx - minx) / reso)
    iy = round((sy - miny) / reso)
    gix = round((gx - minx) / reso)
    giy = round((gy - miny) / reso)
    
    # Variables to track local minima
    stuck_count = 0
    last_d = d
    
    # Add momentum to avoid oscillation and help escape local minima
    prev_dx, prev_dy = 0, 0
    momentum = 0.1  # Momentum coefficient
    
    if show_animation:
        plt.axis([minx, minx + len(pmap) * reso, miny, miny + len(pmap[0]) * reso])
        plt.title('Potential Field Planning with Dynamic Obstacles')
        draw_heatmap(pmap, minx, miny, maxx, maxy)
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(ix, iy, "*k")
        plt.plot(gix, giy, "*m")
    
    while d >= reso and time < max_time:
        # Get current obstacle positions
        ox = [obs.get_centroid()[0] for obs in obs]
        oy = [obs.get_centroid()[1] for obs in obs]
        
        # Recalculate potential field with updated obstacle positions
        pmap, minx, miny, maxx, maxy = calc_potential_field(gx, gy, obs, reso, rr, x, y)
        
        # Current position on grid
        ix = round((x - minx) / reso)
        iy = round((y - miny) / reso)
        
        # Detect if we're stuck in a local minimum
        if abs(d - last_d) < reso * 0.1:
            stuck_count += 1
        else:
            stuck_count = 0
        last_d = d
        
        # Find next position (same as original algorithm)
        minp = float("inf")
        minix, miniy = -1, -1
        
        # If stuck, add random movement to escape local minimum
        if stuck_count > 3:
            print(f"Attempting to escape local minimum at ({ix},{iy})")
            # Add random direction to escape
            rand_motion = np.random.randint(0, len(motion))
            inx = int(ix + motion[rand_motion][0])
            iny = int(iy + motion[rand_motion][1])
            if 0 <= inx < len(pmap) and 0 <= iny < len(pmap[0]):
                minix, miniy = inx, iny
                stuck_count = 0
        else:
            # Normal path planning
            for i, _ in enumerate(motion):
                inx = int(ix + motion[i][0])
                iny = int(iy + motion[i][1])
                
                if inx >= len(pmap) or iny >= len(pmap[0]) or inx < 0 or iny < 0:
                    p = float("inf")  # outside area
                else:
                    p = pmap[inx][iny]
                    
                    # Add momentum effect - prefer continuing in the same direction
                    dx = motion[i][0]
                    dy = motion[i][1]
                    
                    # Calculate direction similarity with previous movement
                    direction_similarity = dx * prev_dx + dy * prev_dy
                    
                    # Apply momentum by reducing potential for similar directions
                    if direction_similarity > 0:
                        momentum_discount = momentum * direction_similarity
                        p -= momentum_discount
                    
                if minp > p:
                    minp = p
                    minix = inx
                    miniy = iny
        
        # Update momentum
        if minix != -1 and miniy != -1:
            prev_dx = minix - ix
            prev_dy = miniy - iy
        
        # Update position
        ix = minix
        iy = miniy
        x = ix * reso + minx
        y = iy * reso + miny
        
        # Update path
        rx.append(x)
        ry.append(y)
        
        # Update obstacle positions
        for ob in obs:
            if isinstance(ob, DynamicObstacle):
                ob.move(dt)
        
        # Check oscillation - don't break, just warn
        if oscillations_detection(previous_ids, ix, iy):
            print(f"Warning: Oscillation detected at ({ix},{iy})")
            # Clear the previous positions to allow continued movement
            previous_ids.clear()
        
        # Update distance to goal
        d = np.hypot(gx - x, gy - y)
        
        # Update simulation time
        time += dt
        
        # Visualization
        if show_animation:
            plt.cla()
            # Calculate heatmap extents
            xw = len(pmap)
            yw = len(pmap[0])
            maxx = minx + xw * reso
            maxy = miny + yw * reso
            draw_heatmap(pmap, minx, miny, maxx, maxy)
            
            # Draw start and goal
            plt.plot(sx, sy, "xr")  # start
            plt.plot(gx, gy, "xb")  # goal
            
            # Draw robot path
            plt.plot(rx, ry, "-r")
            
            # Draw current robot position
            circle = plt.Circle((x, y), rr/15, color='g')
            plt.gca().add_patch(circle)
            
            plt.grid(True)
            plt.axis("equal")
            plt.title(f"Time: {time:.1f}s")
            # Display robot and environment parameters
            plt.figtext(0.02, 0.02, 
                  f"Robot radius: {rr:.1f}m\n"
                  f"Attractive gain (KP): {KP}\n"
                  f"Repulsive gain (ETA): {ETA}\n"
                  f"Resolution: {reso:.2f}m", 
                  bbox=dict(facecolor='white', alpha=0.5))

            # Draw obstacles with proper shapes
            for ob in obs:
              shape = ob.shape
              if isinstance(shape, Circle):
                centroid = shape.get_centroid()
                circle = plt.Circle((centroid[0], centroid[1]), shape.radius,
                          color='k', fill=True, alpha=0.6)
                plt.gca().add_patch(circle)
              elif isinstance(shape, Rectangle):
                rect = plt.Rectangle((shape.get_centroid()[0] - shape.width/2, shape.get_centroid()[1] - shape.height/2), 
                           shape.width, shape.height, 
                           color='k', fill=True, alpha=0.6)
                plt.gca().add_patch(rect)
              
              # Add velocity vector for dynamic obstacles
              if isinstance(ob, DynamicObstacle):
                plt.arrow(shape.get_centroid()[0], shape.get_centroid()[1], ob.vx, ob.vy, 
                     head_width=0.5, head_length=0.7, fc='r', ec='r')
            plt.pause(0.01)  # Increased pause time for better visualization
    
    print("Goal!!" if d < reso else f"Failed to reach goal. Final distance: {d:.2f}")
    return rx, ry


def main():
  obs = [
    StaticObstacle(Circle(10, 10, 3)),
    StaticObstacle(Rectangle(20, 10, 3, 3)),
    # DynamicObstacle(Circle(15, 15, 2), -1, 1),
    # DynamicObstacle(Rectangle(25, 15, 2, 2), 0, 2)
  ]

  sx = 0.0  # start x position [m]
  sy = 0.0  # start y position [m]
  gx = 30.0  # goal x position [m]
  gy = 30.0  # goal y position [m]
  reso = 0.5  # potential grid size [m]
  robot_radius = 15.0  # robot radius [m]

  if show_animation:
    plt.grid(True)
    plt.axis("equal")

  # path generation
  _, _ = dynamic_potential_field_planning(sx, sy, gx, gy, obs, reso, robot_radius)

  if show_animation:
    plt.show()

if __name__ == '__main__':
  main()
