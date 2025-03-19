import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import ast
import os
import os.path as path

# Đường dẫn đến file chứa labels của ImageNet
ABS_PATH = path.abspath(__file__)
DIR_NAME = path.dirname(ABS_PATH)
LABELS_DICT_PATH = path.join(DIR_NAME, 'imagenet1000_clsidx_to_labels.txt')
IMAGE_FOLDER = path.join(DIR_NAME, 'assets/images')

class GoogleNet:
    def __init__(self):
        self.model = models.googlenet(pretrained=True)
        self.model.eval()  # Chuyển mô hình sang chế độ đánh giá

        # Tiền xử lý ảnh (resize, normalize)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Get labels
        if os.path.exists(LABELS_DICT_PATH):
            with open(LABELS_DICT_PATH, 'r') as f:
                self.labels_dict = ast.literal_eval(f.read())

    def predict(self, image_path):
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0)  # Thêm batch dimension

        with torch.no_grad():
            output = self.model(input_tensor)

        predicted_class = output.argmax(dim=1).item()

        output = (predicted_class, *self.labels_dict[predicted_class])

        return output

if __name__ == '__main__':
    model = GoogleNet()

    image_paths = os.listdir(IMAGE_FOLDER)

    for image_name in image_paths:
        image_path = os.path.join(IMAGE_FOLDER, image_name)
        if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing: {image_name}")
            output = model.predict(image_path)
            print(output)