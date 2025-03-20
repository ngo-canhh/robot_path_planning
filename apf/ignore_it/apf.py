"""

Potential Field based path planner

author: Atsushi Sakai (@Atsushi_twi)

Ref:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf

"""

from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# Parameters
# 
KP = 5.0  # attractive potential gain
ETA = 100.0  # repulsive potential gain
AREA_WIDTH = 30.0  # potential area width [m]
# the number of previous positions used to check oscillations
OSCILLATIONS_DETECTION_LENGTH = 10

show_animation = True


class DynamicObstacle:
    def __init__(self, x, y, vx=0.0, vy=0.0):
        self.x = x      # x position
        self.y = y      # y position
        self.vx = vx    # velocity in x direction
        self.vy = vy    # velocity in y direction
    
    def update(self, dt):
        # Update position based on velocity and time step
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Optional: Add boundary conditions here if needed
        # self.x = max(min(self.x, max_x), min_x)


def calc_potential_field(gx, gy, ox, oy, reso, rr, sx, sy):
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
            uo = calc_repulsive_potential(x, y, ox, oy, rr)
            uf = ug + uo
            pmap[ix][iy] = uf

    return pmap, minx, miny


def calc_attractive_potential(x, y, gx, gy):
    return 0.5 * KP * np.hypot(x - gx, y - gy)


def calc_repulsive_potential(x, y, ox, oy, rr):
    # search nearest obstacle
    minid = -1
    dmin = float("inf")
    for i, _ in enumerate(ox):
        d = np.hypot(x - ox[i], y - oy[i])
        if dmin >= d:
            dmin = d
            minid = i

    # calc repulsive potential
    dq = np.hypot(x - ox[minid], y - oy[minid])

    if dq <= rr:
        if dq <= 0.1:
            dq = 0.1

        return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
    else:
        return 0.0


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


def potential_field_planning(sx, sy, gx, gy, ox, oy, reso, rr):

    # calc potential field
    pmap, minx, miny = calc_potential_field(gx, gy, ox, oy, reso, rr, sx, sy)

    # search path
    d = np.hypot(sx - gx, sy - gy)
    ix = round((sx - minx) / reso)
    iy = round((sy - miny) / reso)
    gix = round((gx - minx) / reso)
    giy = round((gy - miny) / reso)

    if show_animation:
        draw_heatmap(pmap)
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(ix, iy, "*k")
        plt.plot(gix, giy, "*m")

    rx, ry = [sx], [sy]
    motion = get_motion_model()
    previous_ids = deque()

    while d >= reso:
        minp = float("inf")
        minix, miniy = -1, -1
        for i, _ in enumerate(motion):
            inx = int(ix + motion[i][0])
            iny = int(iy + motion[i][1])
            if inx >= len(pmap) or iny >= len(pmap[0]) or inx < 0 or iny < 0:
                p = float("inf")  # outside area
                print("outside potential!")
            else:
                p = pmap[inx][iny]
            if minp > p:
                minp = p
                minix = inx
                miniy = iny
        ix = minix
        iy = miniy
        xp = ix * reso + minx
        yp = iy * reso + miny
        d = np.hypot(gx - xp, gy - yp)
        rx.append(xp)
        ry.append(yp)

        if (oscillations_detection(previous_ids, ix, iy)):
            print("Oscillation detected at ({},{})!".format(ix, iy))
            break

        if show_animation:
            plt.plot(ix, iy, ".r")
            plt.pause(0.01)

    print("Goal!!")

    return rx, ry

def draw_heatmap(data):
    data = np.array(data).T
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)

# Add a random walk component to escape local minima 
def dynamic_potential_field_planning(sx, sy, gx, gy, obstacles, reso, rr, dt=0.1, max_time=300.0):
    """
    Path planning with dynamic obstacles using potential field
    """
    # Initial setup similar to original function
    x, y = sx, sy
    rx, ry = [sx], [sy]
    
    # Calculate initial potential field
    ox = [obs.x for obs in obstacles]
    oy = [obs.y for obs in obstacles]
    pmap, minx, miny = calc_potential_field(gx, gy, ox, oy, reso, rr, x, y)
    
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
    
    if show_animation:
        plt.axis([minx, minx + len(pmap) * reso, miny, miny + len(pmap[0]) * reso])
        plt.title('Potential Field Planning with Dynamic Obstacles')
        draw_heatmap_new(pmap)
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(ix, iy, "*k")
        plt.plot(gix, giy, "*m")
    
    while d >= reso and time < max_time:
        # Get current obstacle positions
        ox = [obs.x for obs in obstacles]
        oy = [obs.y for obs in obstacles]
        
        # Recalculate potential field with updated obstacle positions
        pmap, minx, miny = calc_potential_field(gx, gy, ox, oy, reso, rr, x, y)
        
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
        if stuck_count > 10:
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
                if minp > p:
                    minp = p
                    minix = inx
                    miniy = iny
        
        # Update position
        ix = minix
        iy = miniy
        x = ix * reso + minx
        y = iy * reso + miny
        
        # Update path
        rx.append(x)
        ry.append(y)
        
        # Update obstacle positions
        for obs in obstacles:
            obs.update(dt)
        
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
            draw_heatmap_new(pmap, minx, miny, maxx, maxy)
            
            # Draw start and goal
            plt.plot(sx, sy, "xr")  # start
            plt.plot(gx, gy, "xb")  # goal
            
            # Draw robot path
            plt.plot(rx, ry, "-r")
            
            # Draw current robot position
            circle = plt.Circle((x, y), rr/15, color='g')
            plt.gca().add_patch(circle)
            
            # # Draw obstacles
            # for i, (ox_i, oy_i) in enumerate(zip(ox, oy)):
            #     obx = ox_i
            #     oby = oy_i
            #     circle = plt.Circle((obx, oby), rr/5, color='k')
            #     plt.gca().add_patch(circle)
            
            plt.grid(True)
            plt.axis("equal")
            plt.title(f"Time: {time:.1f}s")
            plt.pause(0.01)  # Increased pause time for better visualization
    
    print("Goal!!" if d < reso else f"Failed to reach goal. Final distance: {d:.2f}")
    return rx, ry


def draw_heatmap_new(data, minx=None, miny=None, maxx=None, maxy=None):
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

# def main():
#     print("potential_field_planning start")

#     sx = 0.0  # start x position [m]
#     sy = 10.0  # start y positon [m]
#     gx = 30.0  # goal x position [m]
#     gy = 30.0  # goal y position [m]
#     grid_size = 0.5  # potential grid size [m]
#     robot_radius = 5.0  # robot radius [m]

#     ox = [15.0, 5.0, 20.0, 25.0]  # obstacle x position list [m]
#     oy = [25.0, 15.0, 26.0, 25.0]  # obstacle y position list [m]

#     if show_animation:
#         plt.grid(True)
#         plt.axis("equal")

#     # path generation
#     _, _ = potential_field_planning(
#         sx, sy, gx, gy, ox, oy, grid_size, robot_radius)

#     if show_animation:
#         plt.show()


def main():
    print("Dynamic potential field planning start")

    sx = 0.0   # start x position [m]
    sy = 10.0  # start y positon [m]
    gx = 30.0  # goal x position [m]
    gy = 30.0  # goal y position [m]
    grid_size = 0.5  # potential grid size [m]
    robot_radius = 5.0  # robot radius [m]

    # Create dynamic obstacles with positions and velocities
    obstacles = [
        DynamicObstacle(15.0, 25.0, vx=0.0, vy=-1.4),  # Moving down
        DynamicObstacle(5.0, 15.0, vx=1.0, vy=0.0),    # Moving right
        DynamicObstacle(20.0, 26.0, vx=0.0, vy=0.0),   # Stationary
        DynamicObstacle(25.0, 25.0, vx=-0.2, vy=-0.1)   # Moving up-left
    ]

    if show_animation:
      plt.figure(figsize=(10, 8))
      plt.grid(True)
      plt.axis("equal")

    # path generation with dynamic obstacles
    rx, ry = dynamic_potential_field_planning(
        sx, sy, gx, gy, obstacles, grid_size, robot_radius)

    if show_animation:
        plt.show()

if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")
