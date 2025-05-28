import numpy as np
from components.obstacle import Obstacle, ObstacleType

# --- WaitingRule ---

class WaitingRule:
    def __init__(self, robot_radius, safety_margin=20, prediction_horizon=15, time_step=0.5):
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.prediction_horizon = prediction_horizon
        self.time_step = time_step
        # Lr in the paper - robot length
        self.robot_length = robot_radius * 2
        self.emergency_stop_distance = robot_radius * 3

    def check_static_collision(self, robot_x, robot_y, robot_velocity, robot_orientation, static_obstacles: list):
        """
        Kiểm tra va chạm với vật cản tĩnh phía trước robot
        """
        if robot_velocity < 1e-3:  # Nếu robot đã dừng, không cần kiểm tra
            return False
            
        # Hướng di chuyển của robot
        robot_direction = np.array([np.cos(robot_orientation), np.sin(robot_orientation)])
        
        # Kiểm tra các vật cản tĩnh phía trước
        look_ahead_distance = max(self.robot_radius * 5, robot_velocity * 2)
        
        for obstacle in static_obstacles:
            # Lấy vị trí vật cản
            obstacle_x, obstacle_y = obstacle.get_position()
            obstacle_pos = np.array([obstacle_x, obstacle_y])
            robot_pos = np.array([robot_x, robot_y])
            
            # Vector từ robot đến vật cản
            vec_to_obs = obstacle_pos - robot_pos
            distance = np.linalg.norm(vec_to_obs)
            
            # Nếu vật cản quá gần, dừng ngay lập tức
            if distance < self.emergency_stop_distance + obstacle.shape.get_effective_radius():
                return True
                
            # Kiểm tra nếu vật cản nằm trong hướng di chuyển
            forward_projection = np.dot(vec_to_obs, robot_direction)
            
            # Chỉ xét vật cản phía trước robot (forward_projection > 0) và trong tầm nhìn
            if 0 < forward_projection < look_ahead_distance:
                # Tính khoảng cách từ vật cản đến đường đi của robot
                lateral_distance = np.linalg.norm(vec_to_obs - forward_projection * robot_direction)
                
                # Nếu vật cản nằm trong đường đi (tính cả bán kính an toàn)
                if lateral_distance < self.robot_radius + self.safety_margin + obstacle.shape.get_effective_radius():
                    return True
        
        return False

    def check_dynamic_collisions(self, robot_x, robot_y, robot_velocity, robot_orientation, dynamic_obstacles: list):
        """
        Implements the waiting rule as described in the paper.
        Returns list of (obstacle, time_to_collision, deceleration) tuples for obstacles that require waiting.
        """
        if robot_velocity < 1e-3:  # If robot is already stopped, no need to check
            return []
            
        predicted_collisions = []
        robot_pos = np.array([robot_x, robot_y])
        robot_direction = np.array([np.cos(robot_orientation), np.sin(robot_orientation)])
        
        # Kiểm tra va chạm tức thời
        for obstacle in dynamic_obstacles:
            if obstacle.type != ObstacleType.DYNAMIC:
                continue
                
            # Kiểm tra khoảng cách hiện tại
            obstacle_x, obstacle_y = obstacle.get_position()
            obstacle_pos = np.array([obstacle_x, obstacle_y])
            
            current_distance = np.linalg.norm(robot_pos - obstacle_pos)
            
            # Nếu vật cản quá gần, cần dừng ngay lập tức
            if current_distance < self.robot_radius + obstacle.shape.get_effective_radius() + self.safety_margin * 0.5:
                # Trả về deceleration = robot_velocity/time_step khi cần dừng khẩn cấp
                emergency_decel = robot_velocity / self.time_step
                predicted_collisions.append((obstacle, 0, emergency_decel))
                continue
        
        # Kiểm tra va chạm với vật cản động theo waiting rule
        for obstacle in dynamic_obstacles:
            # Đảm bảo đây là vật cản động và đang di chuyển
            if obstacle.type != ObstacleType.DYNAMIC or obstacle.velocity < 1e-3:
                continue
                
            # Lấy thông số vật cản
            obstacle_x, obstacle_y = obstacle.get_position()
            obstacle_pos = np.array([obstacle_x, obstacle_y])
            obstacle_velocity = obstacle.velocity
            obstacle_direction = np.array(obstacle.direction)  # Đơn vị vector
            
            # Tính góc alpha giữa hướng di chuyển của robot và vật cản (α = θr - θo)
            obstacle_orientation = np.arctan2(obstacle_direction[1], obstacle_direction[0])
            
            # Tính α theo công thức (24)
            alpha = robot_orientation - obstacle_orientation
            # Chuẩn hóa alpha về khoảng [-π, π]
            while alpha > np.pi: alpha -= 2 * np.pi
            while alpha < -np.pi: alpha += 2 * np.pi
            
            # Tính khoảng cách d giữa robot và vật cản
            d = np.linalg.norm(robot_pos - obstacle_pos)
            
            # Xử lý 3 trường hợp góc theo paper: zero, acute, obtuse
            
            # Trường hợp 1: Góc zero (song song)
            if abs(alpha) < 1e-6:
                # Theo paper, song song sẽ không va chạm
                continue
                
            # Trường hợp 2: Góc obtuse (góc tù > 90 độ)
            if abs(alpha) > np.pi/2:
                # Theo paper, góc obtuse không có giao điểm
                continue
                
            # Trường hợp 3: Góc acute (góc nhọn < 90 độ)
            # Tính cos(alpha) trực tiếp từ tích vô hướng
            cos_alpha = np.dot(robot_direction, obstacle_direction)
            
            # Áp dụng điều kiện va chạm từ tam giác (23)
            # vr*t + v0*t > d
            if robot_velocity + obstacle_velocity <= d:
                continue  # Không thể xảy ra va chạm
                
            # Tính thời gian tới điểm va chạm (t0)
            # Từ paper: t0 = d / (vr + v0)
            time_to_collision = d / (robot_velocity + obstacle_velocity)
            
            # Tính khoảng cách robot di chuyển (Sr = vr*t0 - Lr) theo công thức (26)
            distance_to_travel = robot_velocity * time_to_collision - self.robot_length
            
            # Đảm bảo khoảng cách là dương (robot cần giảm tốc)
            if distance_to_travel <= 0:
                # Robot quá gần để dừng kịp, cần hành động ngay
                emergency_decel = robot_velocity / self.time_step
                predicted_collisions.append((obstacle, 0, emergency_decel))
                continue
                
            # Tính toán gia tốc giảm tốc cần thiết theo công thức (27): a = vr²/Sr
            required_deceleration = (robot_velocity**2) / distance_to_travel
            
            # Giảm thời gian dự đoán để phản ứng sớm hơn
            adjusted_time = time_to_collision * 0.8  # Giảm 20% thời gian để an toàn hơn
            
            predicted_collisions.append((obstacle, adjusted_time, required_deceleration))
            
        return predicted_collisions

    def should_wait(self, predicted_collisions):
        """
        Xác định xem robot có nên chờ dựa trên các va chạm dự đoán không.
        Returns: (should_wait: bool, deceleration: float)
        """
        if not predicted_collisions:
            return False, 0.0
            
        # Tìm va chạm gần nhất và gia tốc giảm tốc cần thiết
        closest_collision = min(predicted_collisions, key=lambda x: x[1])
        _, time_to_collision, deceleration = closest_collision
        
        # Đảm bảo luôn trả về giá trị deceleration hợp lệ
        if deceleration is None or deceleration < 0:
            # Nếu không có giá trị giảm tốc hợp lệ, tính toán một giá trị mặc định
            deceleration = 2.0  # Giá trị mặc định
            
        # Nếu va chạm sắp xảy ra (trong bước thời gian tiếp theo)
        if time_to_collision <= self.time_step * 2:  # Phản ứng sớm hơn
            return True, deceleration
            
        # Nếu thời gian va chạm ở trong tầm dự đoán
        if time_to_collision <= self.prediction_horizon:
            return True, deceleration
            
        # Nếu va chạm xa hơn, không cần chờ
        return False, 0.0