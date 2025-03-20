from abc import ABC, abstractmethod
from shape import Shape, Circle, Rectangle
from ggnet import GoogleNet
import os
import os.path as path
import ast

class Obstacle(ABC):
    """
    Abstract base class for obstacles.
    """
    @abstractmethod
    def draw(self):
        pass

    @abstractmethod
    def get_min_distance(self, x, y):
        pass

    @abstractmethod
    def get_centroid(self):
        pass


class StaticObstacle(Obstacle):
    def __init__(self, shape: Shape):
        super().__init__()
        self.shape = shape

    def draw(self):
        pass

    def get_min_distance(self, x, y):
        return self.shape.get_min_distance(x,y)

    def get_centroid(self):
        return self.shape.get_centroid()


class DynamicObstacle(Obstacle):
    def __init__(self, shape: Shape, vx, vy):
        super().__init__()
        self.shape = shape
        self.vx = vx
        self.vy = vy

    def draw(self):
        pass

    def get_min_distance(self, x, y):
        return self.shape.get_min_distance(x, y)

    def get_centroid(self):
        return self.shape.get_centroid()

    def move(self, dt):
        self.shape.x += self.vx * dt
        self.shape.y += self.vy * dt

class ObstacleDetector:
    """
    Phát hiện chướng ngại vật từ hình ảnh sử dụng GoogleNet đã cài đặt
    """
    def __init__(self):
        # Khởi tạo GoogleNet từ class đã có
        self.googlenet = GoogleNet()
        
        # Đọc mapping từ file
        self.obstacle_type_mapping = self._load_mapping()
        
        # Mặc định cho các lớp chưa được định nghĩa
        self.default_type = 'static'

    def _load_mapping(self):
        """
        Đọc mapping từ class index sang dynamic/static từ file imagenet1000_clsidx_to_labels.txt
        """
        mapping = {}
        labels_path = path.join(path.dirname(path.abspath(__file__)), 'imagenet1000_clsidx_to_labels.txt')
        
        try:
            with open(labels_path, 'r') as f:
                content = f.read()
                data = ast.literal_eval(content)
                
                # Data có dạng {0: ('tench, Tinca tinca', 'dynamic'), ...}
                for key, value in data.items():
                    if isinstance(value, tuple) and len(value) >= 2:
                        # Phần tử thứ hai của tuple chứa 'dynamic' hoặc 'static'
                        mapping[key] = value[1]
        except Exception as e:
            print(f"Error loading obstacle type mapping: {e}")
        
        return mapping

    def detect_from_image(self, image_path, position=(0, 0), size=(5, 5), default_velocity=(1, 1)):
        """
        Phát hiện chướng ngại vật từ hình ảnh và tạo đối tượng chướng ngại vật
        """
        try:
            # Dự đoán lớp đối tượng bằng GoogleNet
            pred_class, label, type_info = self.googlenet.predict(image_path)
            
            # Xác định loại chướng ngại vật từ kết quả dự đoán
            if isinstance(type_info, str) and type_info in ['dynamic', 'static']:
                obstacle_type = type_info
            else:
                obstacle_type = self.obstacle_type_mapping.get(pred_class, self.default_type)
            
            # Tạo đối tượng chướng ngại vật dựa trên loại
            x, y = position
            width, height = size
            
            if obstacle_type == 'dynamic':
                # Đối với chướng ngại vật động, sử dụng hình tròn
                radius = min(width, height) / 2
                shape = Circle(x, y, radius)
                vx, vy = default_velocity
                obstacle = DynamicObstacle(shape, vx=vx, vy=vy)
            else:
                # Đối với chướng ngại vật tĩnh, sử dụng hình chữ nhật
                shape = Rectangle(x, y, width, height)
                obstacle = StaticObstacle(shape)
            
            # Store image path with the obstacle
            obstacle.image_path = image_path
            return obstacle
                
        except Exception as e:
            print(f"Error detecting obstacle from image {path.basename(image_path)}: {e}")
            # Trả về chướng ngại vật mặc định nếu có lỗi
            shape = Rectangle(position[0], position[1], size[0], size[1])
            obstacle = StaticObstacle(shape)
            obstacle.image_path = None
            return obstacle

    def detect_obstacles_from_folder(self, positions=None, sizes=None, velocities=None):
        """
        Phát hiện chướng ngại vật từ thư mục hình ảnh assets/images
        """
        # Đường dẫn đến thư mục hình ảnh
        image_folder = path.join(path.dirname(path.abspath(__file__)), 'assets/images')
        
        # Thông số mặc định
        default_position = (0, 0)
        default_size = (5, 5)
        default_velocity = (1, 1)
        
        # Khởi tạo danh sách chướng ngại vật
        static_obstacles = []
        dynamic_obstacles = []
        
        # Nếu positions, sizes hoặc velocities là None, khởi tạo chúng là dict rỗng
        if positions is None:
            positions = {}
        if sizes is None:
            sizes = {}
        if velocities is None:
            velocities = {}
        
        # Lấy danh sách các file hình ảnh
        image_files = [f for f in os.listdir(image_folder) 
                     if path.isfile(path.join(image_folder, f)) and 
                     f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Found {len(image_files)} images in {image_folder}")
        
        # Xử lý từng file hình ảnh
        for image_file in image_files:
            image_path = path.join(image_folder, image_file)
            
            # Lấy thông số cho hình ảnh này
            position = positions.get(image_file, default_position)
            size = sizes.get(image_file, default_size)
            velocity = velocities.get(image_file, default_velocity)
            
            print(f"Processing {image_file} at position {position}, size {size}")
            
            # Determine obstacle type from file name (simplification)
            obstacle_type = 'dynamic' if image_file.lower() in ['cat.png', 'dog.png'] else 'static'
            
            # Create obstacle directly based on type and store image path
            if obstacle_type == 'dynamic':
                x, y = position
                width, height = size
                shape = Rectangle(x, y, width, height)  # Or Circle if preferred
                vx, vy = velocity
                obstacle = DynamicObstacle(shape, vx=vx, vy=vy)
                obstacle.image_path = image_path
                dynamic_obstacles.append(obstacle)
                print(f"  -> Dynamic obstacle created with image: {image_file}")
            else:
                x, y = position
                width, height = size
                shape = Rectangle(x, y, width, height)
                obstacle = StaticObstacle(shape)
                obstacle.image_path = image_path
                static_obstacles.append(obstacle)
                print(f"  -> Static obstacle created with image: {image_file}")
                
        return static_obstacles, dynamic_obstacles


