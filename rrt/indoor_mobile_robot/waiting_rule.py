import math

class WaitingRule:
    """
    Improved Waiting Rule class to avoid dynamic obstacles, 
    implementing formulas from section 3.2.2 of the paper.
    """
    def __init__(self, robot_speed, robot_length=0.5, reaction_time=0.1):
        """
        Initialize WaitingRule with robot parameters.

        Args:
            robot_speed (float): Robot's normal speed.
            robot_length (float): Robot's length (Lr in paper).
            reaction_time (float): Robot's reaction time (to in paper - not explicitly given, assuming a value).
        """
        self.robot_speed = robot_speed
        self.robot_length = robot_length
        self.reaction_time = reaction_time # Reaction time to obstacle

    def calculate_waiting_time(self, robot_pos, robot_orientation, dynamic_obstacle):
        """
        Calculate waiting time needed to avoid dynamic obstacle, 
        implementing formulas from section 3.2.2 of the paper.
        """
        obstacle_pos = (dynamic_obstacle.shape.x, dynamic_obstacle.shape.y)
        obstacle_velocity = dynamic_obstacle.vx, dynamic_obstacle.vy

        robot_x, robot_y = robot_pos
        obstacle_x, obstacle_y = obstacle_pos
        obstacle_vx, obstacle_vy = obstacle_velocity

        vr = self.robot_speed # Robot speed
        vo = math.sqrt(obstacle_vx**2 + obstacle_vy**2) # Obstacle speed magnitude
        d = math.sqrt((obstacle_x - robot_x)**2 + (obstacle_y - robot_y)**2) # Distance between robot and obstacle

        if vr == 0 and vo == 0: # Both robot and obstacle are stationary
            return 0

        # 1. Calculate robot and obstacle orientation (theta and theta_o)
        theta = robot_orientation # Robot orientation is directly given
        theta_o = math.atan2(obstacle_vy, obstacle_vx) if vo != 0 else 0 # Obstacle orientation


        # 2. Calculate alpha angle: α = θ - θo (Eq. 24)
        alpha = theta - theta_o

        # 3. Calculate time to collision tc using Eq. 25 (solve quadratic equation for t)
        # cos α = (vo*t)^2 + (vr*t)^2 - d^2 / (2*vo*vr*t^2)
        # (vo^2 + vr^2 - 2*vo*vr*cos(alpha)) * t^2 = d^2
        A = vo**2 + vr**2 - 2*vo*vr*math.cos(alpha)
        B = 0 # No linear term
        C = -d**2

        if abs(A) < 1e-6: # A is close to zero, avoid division by zero or handle linear case if needed
            if abs(B) < 1e-6:
                collision_time = float('inf') # No solution or infinite solutions
            else:
                collision_time = -C / B # Linear case, if applicable
        else:
            discriminant = B**2 - 4*A*C
            if discriminant < 0:
                collision_time = float('inf') # No real solution, no collision
            else:
                t1 = (-B + math.sqrt(discriminant)) / (2*A)
                t2 = (-B - math.sqrt(discriminant)) / (2*A)
                collision_time = min(t1, t2) if max(t1, t2) >=0 else float('inf') # Take smallest positive root


        if collision_time == float('inf') or collision_time <= 0:
            return 0 # No collision or collision in the past/immediate future

        # 4. Calculate stopping distance Sr using Eq. 26: Sr = vr*to - Lr (using reaction_time for to)
        Sr = vr * self.reaction_time - self.robot_length # Using reaction_time and robot_length

        # 5. Calculate deceleration a using Eq. 27: vr - a*(Sr/vr) = 0  => a = vr^2 / Sr
        if Sr > 0:
            deceleration = (vr**2) / Sr
        else:
            deceleration = 0


        # 6. Calculate actual stopping time (time to decelerate to 0) - not directly from paper formulas, using kinematics
        stopping_time = vr / deceleration if deceleration > 0 else 0 # t = v/a


        # Total waiting time: reaction time + stopping time + a small safety margin
        waiting_time = self.reaction_time + stopping_time + 0.2 # Added reaction time and safety margin

        # Print debugging info
        print(f"Alpha: {math.degrees(alpha):.2f} deg, Collision Time (tc): {collision_time:.2f} s")
        print(f"Stopping Distance (Sr): {Sr:.2f} m, Deceleration (a): {deceleration:.2f} m/s^2")
        print(f"Waiting Time: {waiting_time:.2f} s")


        return max(0, waiting_time) # Ensure waiting time is not negative


    def is_safe_distance(self, robot_pos, dynamic_obstacle, safe_distance):
        """Check safe distance (keep as before)."""
        obstacle_pos = (dynamic_obstacle.shape.x, dynamic_obstacle.shape.y)
        distance = math.sqrt((robot_pos[0] - obstacle_pos[0])**2 + (robot_pos[1] - obstacle_pos[1])**2)
        return distance >= safe_distance