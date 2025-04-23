import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import ast
import os
import os.path as path
import random

ABS_PATH = path.abspath(__file__)
SCRIPT_DIR = path.dirname(ABS_PATH)
PROJECT_ROOT = path.dirname(SCRIPT_DIR) 

LABELS_DICT_PATH = path.join(PROJECT_ROOT, 'assets', 'obstacle_type_label.txt') 
IMAGE_FOLDER = path.join(PROJECT_ROOT, 'assets', 'images') 

class GoogleNet:
    def __init__(self):
        self.model = models.googlenet(pretrained=True)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if os.path.exists(LABELS_DICT_PATH):
            with open(LABELS_DICT_PATH, 'r') as f:
                self.labels_dict = ast.literal_eval(f.read())

    def predict(self, image_path):
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)

        predicted_class = output.argmax(dim=1).item()
        label, is_dynamic = self.labels_dict[predicted_class]
        return predicted_class, label, is_dynamic

def load_random_image():
    """Tải một ảnh ngẫu nhiên từ thư mục assets/images."""
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise ValueError("No images found in the folder 'assets/images'.")
    random_image = random.choice(image_files)
    return os.path.join(IMAGE_FOLDER, random_image)

if __name__ == '__main__':
    model = GoogleNet()
    image_paths = os.listdir(IMAGE_FOLDER)

    for image_name in image_paths:
        image_path = os.path.join(IMAGE_FOLDER, image_name)
        if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing: {image_name}")
            output = model.predict(image_path)
            print(output)