from Obstacle import Obstacle, StaticObstacle, DynamicObstacle
from Shape import Shape, Circle, Rectangle
import numpy as np
from collections import deque
from typing import List
import matplotlib.pyplot as plt
import threading
from queue import Queue

# Parameters
KP = 3.0  # attractive potential gain
ETA = 350.0  # repulsive potential gain
AREA_WIDTH = 10.0  # potential area width [m]
# the number of previous positions used to check oscillations
OSCILLATIONS_DETECTION_LENGTH = 10
show_animation = True
enable_heatmap = False

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

def calc_potential_field_fixed_bounds(gx, gy, obs: List[Obstacle], reso, rr, sx, sy, minx, miny, maxx, maxy):
    """
    Calculate potential field with fixed map boundaries instead of recalculating them
    """
    # Calculate dimensions based on fixed boundaries
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

    return pmap

def draw_heatmap(data, minx=None, miny=None, maxx=None, maxy=None):
    """
    Draw potential field heatmap aligned with world coordinates
    """
    data = np.array(data).T
    
    if minx is not None and miny is not None and maxx is not None and maxy is not None:
        # Set extent to align heatmap with world coordinates
        extent = [minx, maxx, miny, maxy]
        plt.imshow(data, origin='lower', extent=extent, vmax=100.0, cmap=plt.cm.Blues, interpolation='bilinear', aspect='auto')
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

def dynamic_potential_field_planning(sx, sy, gx, gy, obs: List[Obstacle], reso, rr, dt=0.1, max_time=300.0, save_frames=False, save_dir='frames', enable_heatmap=False):
    """
    Path planning with dynamic obstacles using potential field
    """
    # Initial setup similar to original function
    x, y = sx, sy
    rx, ry = [sx], [sy]

    # Add path length tracking
    path_length = 0.0
    
    # Setup for saving frames
    if save_frames:
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        frame_count = 0
        save_thread = start_frame_saver(save_dir)
    
    # Calculate initial potential field and establish fixed map boundaries
    initial_pmap, init_minx, init_miny, init_maxx, init_maxy = calc_potential_field(gx, gy, obs, reso, rr, x, y)
    
    # Store these initial boundaries - they will remain fixed
    fixed_minx = init_minx
    fixed_miny = init_miny
    fixed_maxx = init_maxx
    fixed_maxy = init_maxy
    
    # Set up motion model
    motion = get_motion_model()
    previous_ids = deque()
    
    # Simulation time
    time = 0.0
    
    # Distance to goal
    d = np.hypot(x - gx, y - gy)
    ix = round((sx - fixed_minx) / reso)
    iy = round((sy - fixed_miny) / reso)
    gix = round((gx - fixed_minx) / reso)
    giy = round((gy - fixed_miny) / reso)
    
    # Variables to track local minima
    stuck_count = 0
    last_d = d
    
    if show_animation:
        plt.axis([fixed_minx, fixed_maxx, fixed_miny, fixed_maxy])
        plt.title('Potential Field Planning with Dynamic Obstacles')
        if enable_heatmap:
            draw_heatmap(initial_pmap, fixed_minx, fixed_miny, fixed_maxx, fixed_maxy)
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(sx, sy, "*k")
        plt.plot(gx, gy, "*m")
    
    # Main planning loop
    while d >= reso and time < max_time:
        # Filter out obstacles that are outside the fixed map bounds
        filtered_obs = []
        for ob in obs:
            is_outside = False
            if isinstance(ob, DynamicObstacle):
                shape = ob.shape
                centroid = shape.get_centroid()
                if isinstance(shape, Circle):
                    if (centroid[0] + shape.radius < fixed_minx or 
                        centroid[0] - shape.radius > fixed_maxx or 
                        centroid[1] + shape.radius < fixed_miny or 
                        centroid[1] - shape.radius > fixed_maxy):
                        is_outside = True
                elif isinstance(shape, Rectangle):
                    if (centroid[0] + shape.width/2 < fixed_minx or 
                        centroid[0] - shape.width/2 > fixed_maxx or 
                        centroid[1] + shape.height/2 < fixed_miny or 
                        centroid[1] - shape.height/2 > fixed_maxy):
                        is_outside = True
                
                if is_outside:
                    print(f"Dynamic obstacle at ({centroid[0]:.2f}, {centroid[1]:.2f}) is outside fixed map bounds - ignored in calculations")
                    continue
            
            filtered_obs.append(ob)
        
        # Recalculate potential field using fixed boundaries
        pmap = calc_potential_field_fixed_bounds(gx, gy, filtered_obs, reso, rr, x, y,
                                              fixed_minx, fixed_miny, fixed_maxx, fixed_maxy)
        
        # Current position on grid - use fixed map boundaries
        ix = round((x - fixed_minx) / reso)
        iy = round((y - fixed_miny) / reso)
        
        # Detect if we're stuck in a local minimum
        if abs(d - last_d) < reso * 0.1:
            stuck_count += 1
        else:
            stuck_count = 0
        last_d = d
        
        # Find next position
        minp = float("inf")
        minix, miniy = -1, -1

        # check_dynamic_obstacle = False
        # for ob in filtered_obs:
        #     if isinstance(ob, DynamicObstacle) and ob.get_min_distance(x, y) < rr:
        #         check_dynamic_obstacle = True
        
        # if check_dynamic_obstacle:
        #     minix = ix
        #     miniy = iy
        #     stuck_count = 0
        # el
        if stuck_count > 3:
            print(f"Attempting to escape local minimum at ({ix},{iy})")
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
                if minp > p:
                    minp = p
                    minix = inx
                    miniy = iny
        
        # Update position - using fixed map boundaries
        ix = minix
        iy = miniy
        new_x = ix * reso + fixed_minx
        new_y = iy * reso + fixed_miny
        
        # Calculate segment length and update total path length
        segment_length = np.hypot(new_x - x, new_y - y)
        path_length += segment_length
        
        # Update position
        x = new_x
        y = new_y
        
        # Update path
        rx.append(x)
        ry.append(y)
        
        # Update all obstacle positions (even those outside)
        for ob in obs:
            if isinstance(ob, DynamicObstacle):
                ob.move(dt)
        
        # Check oscillation
        if oscillations_detection(previous_ids, ix, iy):
            print(f"Warning: Oscillation detected at ({ix},{iy})")
            previous_ids.clear()
        
        # Update distance to goal
        d = np.hypot(gx - x, gy - y)
        
        # Update simulation time
        time += dt
        
        # Visualization with fixed map boundaries
        if show_animation:
            plt.cla()
            if enable_heatmap:
                draw_heatmap(pmap, fixed_minx, fixed_miny, fixed_maxx, fixed_maxy)
            
            # Draw start and goal
            plt.plot(sx, sy, "xr")
            plt.plot(gx, gy, "xb")
            
            # Draw robot path
            plt.plot(rx, ry, "-r")
            
            # Draw current robot position
            circle = plt.Circle((x, y), rr/15, color='g')
            plt.gca().add_patch(circle)
            
            plt.grid(True)
            plt.axis([fixed_minx, fixed_maxx, fixed_miny, fixed_maxy])
            plt.title(f"Time: {time:.1f}s")
            
            # Display robot and environment parameters
            plt.figtext(0.02, 0.02, 
                  f"Robot radius: {rr:.1f}m\n"
                  f"Attractive gain (KP): {KP}\n"
                  f"Repulsive gain (ETA): {ETA}\n"
                  f"Resolution: {reso:.2f}m", 
                  bbox=dict(facecolor='white', alpha=0.5))

            # Draw only filtered obstacles
            for ob in filtered_obs:
              shape = ob.shape
              centroid = shape.get_centroid()
                  
              if isinstance(shape, Circle):
                circle = plt.Circle((centroid[0], centroid[1]), shape.radius,
                          color='k', fill=True, alpha=0.6)
                plt.gca().add_patch(circle)
              elif isinstance(shape, Rectangle):
                rect = plt.Rectangle((centroid[0] - shape.width/2, centroid[1] - shape.height/2), 
                            shape.width, shape.height, 
                            color='k', fill=True, alpha=0.6)
                plt.gca().add_patch(rect)
              
              # Add velocity vector for dynamic obstacles
              if isinstance(ob, DynamicObstacle):
                plt.arrow(shape.get_centroid()[0], shape.get_centroid()[1], ob.vx, ob.vy, 
                      head_width=0.5, head_length=0.7, fc='r', ec='r')
            
            # Save the current frame if requested
            if save_frames:
                plt.savefig(f"{save_dir}/frame_{frame_count:04d}.png", dpi=100)
                frame_count += 1
                
            plt.pause(3)
    
    print("Goal!!" if d < reso else f"Failed to reach goal. Final distance: {d:.2f}")
    

    # Save final frame if we're saving frames
    if show_animation and save_frames:
        plt.savefig(f"{save_dir}/frame_{frame_count:04d}.png", dpi=100)
        
    return rx, ry, path_length

# For asynchronous saving
frame_queue = Queue(maxsize=100)
save_thread_active = False

def frame_saver_worker(save_dir):
    """Background thread to save frames without blocking the simulation"""
    global save_thread_active
    
    while save_thread_active or not frame_queue.empty():
        try:
            frame_count, fig = frame_queue.get(timeout=0.1)
            fig.savefig(f"{save_dir}/frame_{frame_count:04d}.png", dpi=100)
            plt.close(fig)  # Close the figure to free memory
            frame_queue.task_done()
        except:
            pass  # Either timeout or queue empty

def start_frame_saver(save_dir):
    """Start the background thread for frame saving"""
    global save_thread_active
    
    save_thread_active = True
    thread = threading.Thread(target=frame_saver_worker, args=(save_dir,))
    thread.daemon = True
    thread.start()
    return thread

def stop_frame_saver():
    """Stop the background thread"""
    global save_thread_active
    save_thread_active = False


def main():
  AREA_WIDTH = 60.0  # Tăng kích thước khu vực tìm kiếm tiềm năng

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

  sx = 0.0  # start x position [m]
  sy = 55.0  # start y position [m]
  gx = AREA_WIDTH - 5.0  # goal x position [m] - điều chỉnh mục tiêu cho bản đồ lớn hơn
  gy = 5.0  # goal y position [m] - điều chỉnh mục tiêu cho bản đồ lớn hơn
  reso = 1  # potential grid size [m]
  robot_radius = 5.0  # robot radius [m]

  if show_animation:
    plt.grid(True)
    plt.axis("equal")
    plt.xlim(0, AREA_WIDTH) # Thiết lập giới hạn trục x cho bản đồ lớn hơn
    plt.ylim(0, AREA_WIDTH) # Thiết lập giới hạn trục y cho bản đồ lớn hơn

  # path generation
  _, _, path_length = dynamic_potential_field_planning(sx, sy, gx, gy, obs, reso, robot_radius, enable_heatmap=enable_heatmap)

  if show_animation:
    # Add path length information to the figure
    plt.figtext(0.5, 0.82, f"Total path length: {path_length:.2f}m", 
                ha="center", fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7))
    plt.show()

if __name__ == '__main__':
    main()
