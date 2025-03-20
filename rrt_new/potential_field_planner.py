import numpy as np
from collections import deque
from typing import List
import matplotlib.pyplot as plt
import threading
from queue import Queue
from obstacle import Obstacle, StaticObstacle, DynamicObstacle
from shape import Shape, Circle, Rectangle

class PotentialFieldPlanner:
    """
    Potential Field Path Planning Class with Dynamic Obstacle Handling.
    """

    # Parameters
    KP = 3.0  # attractive potential gain
    ETA = 350.0  # repulsive potential gain
    AREA_WIDTH = 10.0  # potential area width [m]
    OSCILLATIONS_DETECTION_LENGTH = 10

    def __init__(self,
                 obstacle_list: List[Obstacle],
                 goal: tuple,
                 rand_area: list,
                 reso=1.0,
                 robot_radius=5.0,
                 play_area=None,
                 visualizer=None # RRTVisualizer or similar for visualization
                 ):
        """
        Initialize Potential Field Planner.

        Args:
            obstacle_list (List[Obstacle]): List of obstacle objects.
            goal (tuple): Goal position (gx, gy).
            rand_area (list): Random area (not directly used in APF but kept for consistency).
            reso (float): Potential grid resolution.
            robot_radius (float): Robot radius for repulsive potential.
            play_area (list, optional): Play area bounds. Defaults to None.
            visualizer (RRTVisualizer, optional): Visualizer object. Defaults to None.
        """
        self.obstacle_list = obstacle_list
        self.goal = goal
        self.reso = reso
        self.robot_radius = robot_radius
        self.play_area = play_area
        self.visualizer = visualizer
        self.previous_ids = deque() # For oscillation detection
        self.motion = self.get_motion_model() # Motion model
        self.simulation_time = 0.0 # Simulation time
        self.path_length = 0.0 # Path length


    def calc_attractive_potential(self, x, y, gx, gy):
        """
        Calculate attractive potential.
        """
        return 0.5 * self.KP * np.hypot(x - gx, y - gy)

    def calc_repulsive_potential(self, x, y, obs: List[Obstacle], rr):
        """
        Calculate repulsive potential (nearest obstacle only).
        """
        min_dist = float("inf")
        nearest_obstacle = None
        for ob in obs:
            d = ob.get_min_distance(x, y)
            if d <= 0:
                return float("inf")
            if d <= min_dist:
                min_dist = d
                nearest_obstacle = ob

        if min_dist == float("inf") or min_dist > rr:
            return 0.0
        else:
            return 0.5 * self.ETA * (1.0 / min_dist - 1.0 / rr) ** 2

    def calc_potential_field(self, gx, gy, obs: List[Obstacle], sx, sy):
        """
        Calculate potential field map.
        """
        ox = [ob.get_centroid()[0] for ob in obs]
        oy = [ob.get_centroid()[1] for ob in obs]

        minx = min(min(ox), sx, gx) - self.AREA_WIDTH / 2.0
        miny = min(min(oy), sy, gy) - self.AREA_WIDTH / 2.0
        maxx = max(max(ox), sx, gx) + self.AREA_WIDTH / 2.0
        maxy = max(max(oy), sy, gy) + self.AREA_WIDTH / 2.0
        xw = int(round((maxx - minx) / self.reso))
        yw = int(round((maxy - miny) / self.reso))

        # calc each potential
        pmap = [[0.0 for _ in range(yw)] for _ in range(xw)]

        for ix in range(xw):
            x = ix * self.reso + minx
            for iy in range(yw):
                y = iy * self.reso + miny
                ug = self.calc_attractive_potential(x, y, gx, gy)
                uo = self.calc_repulsive_potential(x, y, obs, self.robot_radius)
                if uo == float("inf"):
                    pmap[ix][iy] = float("inf")
                else:
                    uf = ug + uo
                    pmap[ix][iy] = uf

        return pmap, minx, miny, maxx, maxy

    def calc_potential_field_fixed_bounds(self, gx, gy, obs: List[Obstacle], minx, miny, maxx, maxy):
        """
        Calculate potential field with fixed map boundaries.
        """
        xw = int(round((maxx - minx) / self.reso))
        yw = int(round((maxy - miny) / self.reso))

        # calc each potential
        pmap = [[0.0 for _ in range(yw)] for _ in range(xw)]

        for ix in range(xw):
            x = ix * self.reso + minx
            for iy in range(yw):
                y = iy * self.reso + miny
                ug = self.calc_attractive_potential(x, y, gx, gy)
                uo = self.calc_repulsive_potential(x, y, obs, self.robot_radius)
                if uo == float("inf"):
                    pmap[ix][iy] = float("inf")
                else:
                    uf = ug + uo
                    pmap[ix][iy] = uf

        return pmap

    def draw_heatmap(self, data, minx=None, miny=None, maxx=None, maxy=None):
        """
        Draw potential field heatmap aligned with world coordinates.
        """
        data = np.array(data).T

        if minx is not None and miny is not None and maxx is not None and maxy is not None:
            extent = [minx, maxx, miny, maxy]
            plt.imshow(data, origin='lower', extent=extent, vmax=100.0, cmap=plt.cm.Blues, interpolation='bilinear', aspect='auto')
        else:
            plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)

    def oscillations_detection(self, ix, iy):
        """
        Detect oscillations in path.
        """
        self.previous_ids.append((ix, iy))

        if len(self.previous_ids) > self.OSCILLATIONS_DETECTION_LENGTH:
            self.previous_ids.popleft()

        previous_ids_set = set()
        for index in self.previous_ids:
            if index in previous_ids_set:
                return True
            else:
                previous_ids_set.add(index)
        return False

    def get_motion_model(self):
        """
        Get motion model.
        """
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

    def dynamic_potential_field_planning(self, sx, sy, gx, gy, dt=0.1, max_time=300.0, save_frames=False, save_dir='frames', enable_heatmap=False, show_animation=True):
        """
        Path planning with dynamic obstacles using potential field.
        """
        x, y = sx, sy
        rx, ry = [sx], [sy]
        path_length = 0.0

        if save_frames:
            import os
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            frame_count = 0
            save_thread = self.start_frame_saver(save_dir)

        initial_pmap, init_minx, init_miny, init_maxx, init_maxy = self.calc_potential_field(gx, gy, self.obstacle_list, x, y)

        fixed_minx = init_minx
        fixed_miny = init_miny
        fixed_maxx = init_maxx
        fixed_maxy = init_maxy

        motion = self.get_motion_model()
        previous_ids = deque()

        time = 0.0
        d = np.hypot(x - gx, y - gy)
        ix = round((sx - fixed_minx) / self.reso)
        iy = round((sy - fixed_miny) / self.reso)
        gix = round((gx - fixed_minx) / self.reso)
        giy = round((gy - fixed_miny) / self.reso)

        stuck_count = 0
        last_d = d

        if show_animation and self.visualizer:
            plt.axis([fixed_minx, fixed_maxx, fixed_miny, fixed_maxy])
            plt.title('Potential Field Planning with Dynamic Obstacles')
            if enable_heatmap:
                self.draw_heatmap(initial_pmap, fixed_minx, fixed_miny, fixed_maxx, fixed_maxy)
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(sx, sy, "*k")
            plt.plot(gx, gy, "*m")

        while d >= self.reso and time < max_time:
            filtered_obs = []
            for ob in self.obstacle_list:
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

            pmap = self.calc_potential_field_fixed_bounds(gx, gy, filtered_obs,
                                                      fixed_minx, fixed_miny, fixed_maxx, fixed_maxy)

            ix = round((x - fixed_minx) / self.reso)
            iy = round((y - fixed_miny) / self.reso)

            if abs(d - last_d) < self.reso * 0.1:
                stuck_count += 1
            else:
                stuck_count = 0
            last_d = d

            minp = float("inf")
            minix, miniy = -1, -1

            if stuck_count > 3:
                print(f"Attempting to escape local minimum at ({ix},{iy})")
                rand_motion = np.random.randint(0, len(motion))
                inx = int(ix + motion[rand_motion][0])
                iny = int(iy + motion[rand_motion][1])
                if 0 <= inx < len(pmap) and 0 <= iny < len(pmap[0]):
                    minix, miniy = inx, iny
                    stuck_count = 0
            else:
                for i, _ in enumerate(motion):
                    inx = int(ix + motion[i][0])
                    iny = int(iy + motion[i][1])
                    if inx >= len(pmap) or iny >= len(pmap[0]) or inx < 0 or iny < 0:
                        p = float("inf")
                    else:
                        p = pmap[inx][iny]
                    if minp > p:
                        minp = p
                        minix = inx
                        miniy = iny

            ix = minix
            iy = miniy
            new_x = ix * self.reso + fixed_minx
            new_y = iy * self.reso + fixed_miny

            segment_length = np.hypot(new_x - x, new_y - y)
            self.path_length += segment_length

            x = new_x
            y = new_y

            rx.append(x)
            ry.append(y)

            for ob in self.obstacle_list:
                if isinstance(ob, DynamicObstacle):
                    ob.move(dt)

            if self.oscillations_detection(ix, iy):
                print(f"Warning: Oscillation detected at ({ix},{iy})")
                self.previous_ids.clear()

            d = np.hypot(gx - x, gy - y)
            time += dt
            self.simulation_time = time

            if show_animation and self.visualizer:
                plt.cla()
                if enable_heatmap:
                    self.draw_heatmap(pmap, fixed_minx, fixed_miny, fixed_maxx, fixed_maxy)

                plt.plot(sx, sy, "xr")
                plt.plot(gx, gy, "xb")

                plt.plot(rx, ry, "-r")

                circle = plt.Circle((x, y), self.robot_radius/15, color='g')
                plt.gca().add_patch(circle)

                plt.grid(True)
                plt.axis([fixed_minx, fixed_maxx, fixed_miny, fixed_maxy])
                plt.title(f"Time: {time:.1f}s")

                plt.figtext(0.02, 0.02,
                      f"Robot radius: {self.robot_radius:.1f}m\n"
                      f"Attractive gain (KP): {self.KP}\n"
                      f"Repulsive gain (ETA): {self.ETA}\n"
                      f"Resolution: {self.reso:.2f}m",
                      bbox=dict(facecolor='white', alpha=0.5))

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

                    if isinstance(ob, DynamicObstacle):
                        plt.arrow(shape.get_centroid()[0], shape.get_centroid()[1], ob.vx, ob.vy,
                              head_width=0.5, head_length=0.7, fc='r', ec='r')


                plt.pause(0.01)

        print("Goal!!" if d < self.reso else f"Failed to reach goal. Final distance: {d:.2f}")

        return rx, ry, self.path_length

    # For asynchronous saving (moved inside class)
    frame_queue = Queue(maxsize=100)
    save_thread_active = False

    def frame_saver_worker(self, save_dir):
        """Background thread to save frames without blocking the simulation"""
        global PotentialFieldPlanner  # Access class-level variable

        while PotentialFieldPlanner.save_thread_active or not PotentialFieldPlanner.frame_queue.empty():
            try:
                frame_count, fig = PotentialFieldPlanner.frame_queue.get(timeout=0.1)
                fig.savefig(f"{save_dir}/frame_{frame_count:04d}.png", dpi=100)
                plt.close(fig)  # Close the figure to free memory
                PotentialFieldPlanner.frame_queue.task_done()
            except:
                pass  # Either timeout or queue empty

    def start_frame_saver(self, save_dir):
        """Start the background thread for frame saving"""
        global PotentialFieldPlanner  # Access class-level variable

        PotentialFieldPlanner.save_thread_active = True
        thread = threading.Thread(target=self.frame_saver_worker, args=(save_dir,))
        thread.daemon = True
        thread.start()
        return thread

    def stop_frame_saver(self):
        """Stop the background thread"""
        global PotentialFieldPlanner  # Access class-level variable
        PotentialFieldPlanner.save_thread_active = False