# --- START OF FILE bubble_rebound_v2.py ---

import numpy as np
import math
from components.obstacle import Obstacle, StaticObstacle
from components.shape import Shape, Circle, Rectangle # Giả sử Shape được định nghĩa ở đâu đó

# Thêm import cho matplotlib (chỉ cần thiết nếu sử dụng visualize)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Cảnh báo: matplotlib không được cài đặt. Chức năng trực quan hóa sẽ không khả dụng.")


class BubbleRebound:
    def __init__(self, env_width: float, env_height: float, env_dt: float, num_rays: int, robot_radius: float, sensor_range: float, K_values: list[float] = None):
        """
        Khởi tạo bộ tính toán Bubble Rebound.

        Args:
            env_width: Chiều rộng của môi trường.
            env_height: Chiều cao của môi trường.
            env_dt: Bước thời gian của môi trường/cập nhật bộ điều khiển.
            num_rays: Số lượng tia cảm biến ảo để phát ra.
            robot_radius: Bán kính của robot.
            sensor_range: Phạm vi tối đa của các cảm biến ảo.
            K_values: Danh sách các hằng số Ki cho mỗi tia, kiểm soát ranh giới bong bóng.
                      Nếu None, mặc định là 1.0 cho tất cả các tia.
        """
        if num_rays <= 0:
            raise ValueError("Số lượng tia phải là số dương.")
        if env_dt <= 0:
            print("Cảnh báo: env_dt không dương, sử dụng mặc định 0.1s cho tính toán bong bóng.")
            self.env_dt = 0.1
        else:
            self.env_dt = env_dt

        self.num_rays = num_rays
        self.robot_radius = robot_radius
        self.sensor_range = sensor_range
        self.width = env_width
        self.height = env_height
        # self.angular_step = math.pi / self.num_rays # Giả định góc quét 180 độ
        # Góc quét đầy đủ hơn: 2*pi / num_rays nếu bao phủ 360 độ
        # Giữ nguyên góc quét 180 độ về phía trước:
        self.angular_step = math.pi / (self.num_rays - 1) if self.num_rays > 1 else math.pi
        self.epsilon = 1e-9 # Giá trị nhỏ để tránh chia cho 0

        # Khởi tạo giá trị K (hằng số ranh giới bong bóng cho mỗi tia)
        if K_values is not None:
            if len(K_values) != num_rays:
                raise ValueError(f"Độ dài của K_values phải là {num_rays}")
            self.K = np.array(K_values)
        else:
            self.K = np.ones(num_rays)

        # Tính toán trước các góc tia tương đối so với hướng tiến của robot
        # Các góc nằm trong khoảng từ -pi/2 đến +pi/2 so với hướng tiến
        if self.num_rays > 1:
             self.relative_ray_angles = np.linspace(-math.pi / 2, math.pi / 2, self.num_rays)
        else:
             self.relative_ray_angles = np.array([0.0]) # Chỉ có một tia về phía trước

    def set_K(self, K_values: list[float]):
        """Đặt các hằng số Ki cho tính toán ranh giới bong bóng."""
        if len(K_values) != self.num_rays:
            raise ValueError(f"Độ dài của K_values phải là {self.num_rays}")
        self.K = np.array(K_values)

    def _get_ray_intersection_distance(self, robot_pos: np.ndarray, ray_dir: np.ndarray, obstacles: list[Obstacle]) -> float:
        """Tính toán khoảng cách giao cắt gần nhất cho một tia đơn."""
        closest_t = self.sensor_range

        # 1. Kiểm tra giao cắt với các chướng ngại vật
        for obs in obstacles:
            obs_pos = np.array(obs.get_position())
            # Giả sử intersect_ray nhận vị trí robot và hướng tia
            # và trả về khoảng cách t dọc theo tia từ vị trí robot
            t = obs.shape.intersect_ray(robot_pos, ray_dir, obs_pos[0], obs_pos[1])

            if 0 < t < closest_t: # Kiểm tra 0 < t để tránh giao cắt phía sau robot
                 closest_t = t

        # 2. Kiểm tra giao cắt với ranh giới môi trường
        boundary_t = self.sensor_range
        # Trái (x=0)
        if ray_dir[0] < -self.epsilon:
            t_bound = -robot_pos[0] / ray_dir[0]
            if 0 <= t_bound < boundary_t: boundary_t = t_bound
        # Phải (x=width)
        if ray_dir[0] > self.epsilon:
            t_bound = (self.width - robot_pos[0]) / ray_dir[0]
            if 0 <= t_bound < boundary_t: boundary_t = t_bound
        # Dưới (y=0)
        if ray_dir[1] < -self.epsilon:
            t_bound = -robot_pos[1] / ray_dir[1]
            if 0 <= t_bound < boundary_t: boundary_t = t_bound
        # Trên (y=height)
        if ray_dir[1] > self.epsilon:
            t_bound = (self.height - robot_pos[1]) / ray_dir[1]
            if 0 <= t_bound < boundary_t: boundary_t = t_bound

        # Khoảng cách cuối cùng là minimum của va chạm chướng ngại vật, va chạm ranh giới, hoặc phạm vi cảm biến
        final_t = min(closest_t, boundary_t) # Đã bao gồm sensor_range trong closest_t và boundary_t

        # Đảm bảo khoảng cách không âm
        return max(0.0, final_t)


    def compute_rebound_angle(self, robot_x: float, robot_y: float, robot_orientation: float, robot_velocity: float, obstacles: list[Obstacle]):
        """
        Tính toán góc bật lại dựa trên thuật toán Bubble Rebound.

        Args:
            robot_x, robot_y: Vị trí robot hiện tại.
            robot_orientation: Góc hướng robot hiện tại (theo radian).
            robot_velocity: Vận tốc vô hướng hiện tại của robot.
            obstacles: Danh sách các chướng ngại vật được cảm nhận.

        Returns:
            tuple: (rebound_angle, is_rebound_active, min_distance, measured_distances)
                   rebound_angle (float): Góc bật lại được tính toán (theo radian).
                                           Mặc định là robot_orientation nếu không cần bật lại.
                   is_rebound_active (bool): True nếu có bất kỳ chướng ngại vật nào được phát hiện
                                             trong ranh giới bong bóng của nó.
                   min_distance (float): Khoảng cách đo được ngắn nhất trong số tất cả các tia.
                   measured_distances (np.ndarray): Mảng các khoảng cách đo được cho mỗi tia.
        """
        robot_pos = np.array([robot_x, robot_y])
        sum_weighted_angles = 0.0
        sum_weights = 0.0
        is_rebound_active = False
        min_distance = float('inf')
        measured_distances = np.full(self.num_rays, self.sensor_range) # Khởi tạo mảng khoảng cách

        # Tính toán ranh giới bong bóng cho tất cả các tia trước
        # Định nghĩa trong bài báo: Kᵢ * |V| * Δt. Đây là ngưỡng khoảng cách.
        # Sử dụng abs(robot_velocity) + epsilon để tránh boundary=0 nếu V=0
        bubble_boundaries = self.K * (abs(robot_velocity) + self.epsilon) * self.env_dt

        # Lặp qua từng tia ảo
        for i in range(self.num_rays):
            # Tính góc tuyệt đối của tia trong hệ quy chiếu thế giới
            absolute_ray_angle = robot_orientation + self.relative_ray_angles[i]
            absolute_ray_angle = math.atan2(math.sin(absolute_ray_angle), math.cos(absolute_ray_angle)) # Chuẩn hóa góc

            ray_dir = np.array([math.cos(absolute_ray_angle), math.sin(absolute_ray_angle)])

            # Lấy khoảng cách đo được dọc theo tia này (Di trong bài báo)
            dist = self._get_ray_intersection_distance(robot_pos, ray_dir, obstacles)
            measured_distances[i] = dist # Lưu khoảng cách đo được

            # --- Kiểm tra xem tia này có chạm vào vật gì trong ranh giới bong bóng của nó không ---
            if dist < bubble_boundaries[i]:
                is_rebound_active = True
                # print(f"Ray {i}: Hit within bubble! Dist={dist:.2f}, Boundary={bubble_boundaries[i]:.2f}") # Debug

            # --- Tích lũy cho tính toán góc bật lại (sử dụng TẤT CẢ các tia) ---
            # Trọng số = khoảng cách đo được (Di). Góc = góc tia tuyệt đối (αi).
            # Công thức bài báo: AR = Σ(αi * Di) / Σ(Di)
            # CHÚ Ý QUAN TRỌNG: Bài báo gốc và một số triển khai sử dụng trọng số tỷ lệ nghịch
            # với khoảng cách (ví dụ: 1/Di hoặc (sensor_range - Di)).
            # Sử dụng trọng số = Di trực tiếp như trong mã giả của bài báo có vẻ trực quan hơn
            # cho việc tính góc trung bình có trọng số, nhưng có thể dẫn đến các tia xa hơn
            # có ảnh hưởng lớn hơn. Hãy thử cả hai cách nếu kết quả không như ý.
            # Hiện tại, chúng ta theo mã giả: trọng số = Di.
            weight = 1/ dist
            # Cân nhắc: Nếu dist rất nhỏ (gần va chạm), trọng số này cũng nhỏ.
            # Có lẽ nên dùng (sensor_range - dist) làm trọng số? Hoặc 1/dist?
            # Thử với weight = self.sensor_range - dist (vật cản gần hơn có trọng số lớn hơn)
            # weight = max(0, self.sensor_range - dist) # Đảm bảo trọng số không âm

            # Theo công thức gốc AR = Σ(αi * Di) / Σ(Di):
            sum_weighted_angles += absolute_ray_angle * weight
            sum_weights += weight

            min_distance = min(min_distance, dist)

        # Tính toán góc bật lại cuối cùng (AR)
        if is_rebound_active and sum_weights > self.epsilon:
            # Tính góc trung bình theo atan2 để xử lý wrap-around
            avg_cos = 0.0
            avg_sin = 0.0
            for i in range(self.num_rays):
                absolute_ray_angle = robot_orientation + self.relative_ray_angles[i]
                weight = measured_distances[i] # Theo công thức Σ(αi * Di) / Σ(Di)
                # weight = max(0, self.sensor_range - measured_distances[i]) # Trọng số tỷ lệ nghịch

                # Điều chỉnh trọng số cho các tia trong vùng nguy hiểm (dist < bubble_boundary)
                # Có thể tăng trọng số của các tia này lên đáng kể
                # if measured_distances[i] < bubble_boundaries[i]:
                #      weight *= 5 # Ví dụ: tăng trọng số gấp 5 lần

                avg_cos += math.cos(absolute_ray_angle) * weight
                avg_sin += math.sin(absolute_ray_angle) * weight

            if abs(avg_cos) > self.epsilon or abs(avg_sin) > self.epsilon:
                 rebound_angle = math.atan2(avg_sin/sum_weights, avg_cos/sum_weights)
            else: # Trường hợp tổng trọng số gần như bằng 0
                 rebound_angle = robot_orientation # Không có hướng rõ ràng, giữ nguyên
                 is_rebound_active = False # Coi như không hoạt động nếu không có trọng số

            # print(f"Rebound Active! Angle: {math.degrees(rebound_angle):.1f}") # Debug
        else:
            rebound_angle = robot_orientation
            # print("Rebound NOT Active.") # Debug
            is_rebound_active = False # Đảm bảo là False nếu không bật lại

        # Trả về cả các khoảng cách đo được
        return rebound_angle, is_rebound_active, min_distance, measured_distances


    def visualize(self, ax, robot_x: float, robot_y: float, robot_orientation: float, robot_velocity: float, obstacles: list[Obstacle]):
        """
        Trực quan hóa trạng thái hiện tại của Bubble Rebound trên một Axes matplotlib.

        Args:
            ax: Đối tượng matplotlib.axes.Axes để vẽ lên.
            robot_x, robot_y: Vị trí robot hiện tại.
            robot_orientation: Góc hướng robot hiện tại (theo radian).
            robot_velocity: Vận tốc vô hướng hiện tại của robot.
            obstacles: Danh sách các chướng ngại vật được cảm nhận.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Lỗi: matplotlib không khả dụng. Không thể thực hiện trực quan hóa.")
            return

        robot_pos = np.array([robot_x, robot_y])

        # Tính toán lại các giá trị cần thiết (hoặc nhận chúng làm tham số nếu đã tính toán)
        rebound_angle, is_rebound_active, min_dist, measured_distances = self.compute_rebound_angle(
            robot_x, robot_y, robot_orientation, robot_velocity, obstacles
        )
        bubble_boundaries = self.K * (abs(robot_velocity) + self.epsilon) * self.env_dt

        # --- Vẽ ---
        # 1. Vẽ Robot
        robot_patch = patches.Circle(robot_pos, self.robot_radius, color='blue', alpha=0.5, label='Robot')
        ax.add_patch(robot_patch)
        # Vẽ hướng robot
        ax.quiver(robot_x, robot_y,
                  self.robot_radius * math.cos(robot_orientation),
                  self.robot_radius * math.sin(robot_orientation),
                  angles='xy', scale_units='xy', scale=1, color='black', width=0.005)

        # 2. Vẽ các tia cảm biến và điểm giao cắt
        bubble_points = []
        for i in range(self.num_rays):
            absolute_ray_angle = robot_orientation + self.relative_ray_angles[i]
            absolute_ray_angle = math.atan2(math.sin(absolute_ray_angle), math.cos(absolute_ray_angle))
            ray_dir = np.array([math.cos(absolute_ray_angle), math.sin(absolute_ray_angle)])

            # Điểm cuối tia (tại khoảng cách đo được)
            end_point = robot_pos + ray_dir * measured_distances[i]
            # Điểm ranh giới bong bóng dọc theo tia
            bubble_point = robot_pos + ray_dir * bubble_boundaries[i]
            bubble_points.append(bubble_point)

            # Màu tia dựa trên việc có nằm trong bong bóng không
            ray_color = 'red' if measured_distances[i] < bubble_boundaries[i] else 'gray'
            line_style = '-' if measured_distances[i] < bubble_boundaries[i] else '--'
            alpha = 0.8 if measured_distances[i] < bubble_boundaries[i] else 0.4

            # Vẽ tia (đến điểm giao cắt)
            ax.plot([robot_x, end_point[0]], [robot_y, end_point[1]], color=ray_color, linestyle=line_style, alpha=alpha, linewidth=0.8)
            # Vẽ điểm giao cắt (nếu không phải là sensor_range)
            if measured_distances[i] < self.sensor_range - self.epsilon:
                ax.scatter(end_point[0], end_point[1], color='red', s=10, zorder=5)

            # Vẽ điểm đánh dấu ranh giới bong bóng trên tia (tùy chọn)
            # ax.scatter(bubble_point[0], bubble_point[1], color='orange', s=5, alpha=0.6)

        # 3. Vẽ đường viền bong bóng (nối các điểm bubble_point)
        if len(bubble_points) > 1:
             bubble_poly_points = np.array(bubble_points)
             # Thêm điểm gốc để đóng đa giác nếu cần (không cần thiết nếu chỉ vẽ đường viền)
             # bubble_poly_points = np.vstack([robot_pos, bubble_poly_points, robot_pos]) # Tạo hình quạt
             ax.plot(bubble_poly_points[:, 0], bubble_poly_points[:, 1], color='orange', linestyle=':', linewidth=1, label='Bubble Boundary')

        # 4. Vẽ góc bật lại nếu hoạt động
        if is_rebound_active:
            # Chiều dài mũi tên tỷ lệ với vận tốc hoặc cố định
            arrow_len = self.robot_radius * 2
            ax.quiver(robot_x, robot_y,
                      arrow_len * math.cos(rebound_angle),
                      arrow_len * math.sin(rebound_angle),
                      angles='xy', scale_units='xy', scale=1, color='green', width=0.008, label=f'Rebound ({math.degrees(rebound_angle):.1f} deg)')

        # 5. Thiết lập hiển thị
        ax.set_aspect('equal', adjustable='box')
        # Đặt giới hạn trục nếu cần, ví dụ dựa trên vị trí robot và phạm vi cảm biến
        view_range = self.sensor_range * 1.5
        ax.set_xlim(robot_x - view_range, robot_x + view_range)
        ax.set_ylim(robot_y - view_range, robot_y + view_range)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        title = f"Bubble Rebound Visualization (Rebound: {'Active' if is_rebound_active else 'Inactive'})"
        ax.set_title(title)
        # ax.legend() # Có thể thêm legend nếu cần


# --- END OF FILE bubble_rebound_v2.py ---

# --- START OF EXAMPLE USAGE SCRIPT ---
import numpy as np
import math
import matplotlib.pyplot as plt


# --- Cấu hình mô phỏng ---
ENV_WIDTH = 50.0
ENV_HEIGHT = 50.0
ENV_DT = 0.5 # Time step
NUM_RAYS = 19 # Số tia cảm biến (lẻ để có tia chính giữa)
ROBOT_RADIUS = 1.0
SENSOR_RANGE = 10.0
ROBOT_VELOCITY = 2.0 # Vận tốc robot hiện tại

# --- Tạo đối tượng BubbleRebound ---
bubble_calculator = BubbleRebound(
    env_width=ENV_WIDTH,
    env_height=ENV_HEIGHT,
    env_dt=ENV_DT,
    num_rays=NUM_RAYS,
    robot_radius=ROBOT_RADIUS,
    sensor_range=SENSOR_RANGE,
    K_values=[4] * NUM_RAYS # Sử dụng K=1 mặc định
)

# --- Định nghĩa trạng thái robot và chướng ngại vật ---
robot_x = 10.0
robot_y = 10.0
robot_orientation = math.radians(45) # Hướng 45 độ
# robot_orientation = math.radians(0) # Hướng về bên phải

obstacles = [
    StaticObstacle(15, 11, Circle(radius=1.5)),
    StaticObstacle(12, 18, Circle(radius=2.0)),
    StaticObstacle(5, 5, Circle(radius=1.0)), # Chướng ngại vật gần ranh giới trái/dưới
]

# --- Tạo hình ảnh trực quan ---
fig, ax = plt.subplots(figsize=(8, 8))

# Gọi hàm visualize
bubble_calculator.visualize(ax, robot_x, robot_y, robot_orientation, ROBOT_VELOCITY, obstacles)

# Vẽ thêm chướng ngại vật để đối chiếu
for obs in obstacles:
    if isinstance(obs.shape, Circle):
        obs_patch = patches.Circle(obs.get_position(), obs.shape.radius, color='purple', alpha=0.7, label='Obstacle')
        ax.add_patch(obs_patch)

# Vẽ ranh giới môi trường
ax.plot([0, ENV_WIDTH, ENV_WIDTH, 0, 0], [0, 0, ENV_HEIGHT, ENV_HEIGHT, 0], color='black', linestyle='--', label='Boundary')

# Tinh chỉnh plot
ax.set_xlim(-5, ENV_WIDTH + 5) # Mở rộng view một chút
ax.set_ylim(-5, ENV_HEIGHT + 5)
ax.legend(loc='upper right')
ax.grid(True, linestyle=':', alpha=0.6)

plt.show()

# --- END OF EXAMPLE USAGE SCRIPT ---