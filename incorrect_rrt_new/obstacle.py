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
        self.googlenet = GoogleNet()
        
        self.obstacle_type_mapping = self._load_mapping()
        
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
                
                for key, value in data.items():
                    if isinstance(value, tuple) and len(value) >= 2:
                        mapping[key] = value[1]
        except Exception as e:
            print(f"Error loading obstacle type mapping: {e}")
        
        return mapping

    def detect_from_image(self, image_path, position=(0, 0), size=(5, 5), default_velocity=(1, 1)):
        """
        Phát hiện chướng ngại vật từ hình ảnh và tạo đối tượng chướng ngại vật
        """
        try:
            pred_class, label, type_info = self.googlenet.predict(image_path)
            
            if isinstance(type_info, str) and type_info in ['dynamic', 'static']:
                obstacle_type = type_info
            else:
                obstacle_type = self.obstacle_type_mapping.get(pred_class, self.default_type)
            
            x, y = position
            width, height = size
            
            if obstacle_type == 'dynamic':
                radius = min(width, height) / 2
                shape = Circle(x, y, radius)
                vx, vy = default_velocity
                obstacle = DynamicObstacle(shape, vx=vx, vy=vy)
            else:
                shape = Rectangle(x, y, width, height)
                obstacle = StaticObstacle(shape)
            
            obstacle.image_path = image_path
            return obstacle
                
        except Exception as e:
            print(f"Error detecting obstacle from image {path.basename(image_path)}: {e}")
            shape = Rectangle(position[0], position[1], size[0], size[1])
            obstacle = StaticObstacle(shape)
            obstacle.image_path = None
            return obstacle

    def detect_obstacles_from_folder(self, positions=None, sizes=None, velocities=None):
        """
        Phát hiện chướng ngại vật từ thư mục hình ảnh assets/images
        """
        image_folder = path.join(path.dirname(path.abspath(__file__)), 'assets/images')
        
        default_position = (0, 0)
        default_size = (5, 5)
        default_velocity = (1, 1)
        
        static_obstacles = []
        dynamic_obstacles = []
        
        if positions is None:
            positions = {}
        if sizes is None:
            sizes = {}
        if velocities is None:
            velocities = {}
        
        image_files = [f for f in os.listdir(image_folder) 
                     if path.isfile(path.join(image_folder, f)) and 
                     f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Found {len(image_files)} images in {image_folder}")
        
        for image_file in image_files:
            image_path = path.join(image_folder, image_file)
            
            position = positions.get(image_file, default_position)
            size = sizes.get(image_file, default_size)
            velocity = velocities.get(image_file, default_velocity)
            
            print(f"Processing {image_file} at position {position}, size {size}")
            
            obstacle_type = 'dynamic' if image_file.lower() in ['cat.png', 'dog.png'] else 'static'
            
            if obstacle_type == 'dynamic':
                x, y = position
                width, height = size
                shape = Rectangle(x, y, width, height) 
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


