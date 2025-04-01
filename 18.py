import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import urllib.request

def download_model_if_missing(model_path):
    """Download a default model.pth if it does not exist."""
    if not os.path.exists(model_path):
        print(f"Downloading pretrained model to {model_path}...")
        url = "https://download.pytorch.org/models/deeplabv3_resnet50_coco-586e9e4e.pth"
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")

class SegmentationModel(torch.nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[4] = torch.nn.Conv2d(256, 21, kernel_size=(1, 1))  # Adjust for 21 classes (Pascal VOC)

    def forward(self, x):
        return self.model(x)['out']

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0), np.array(image) / 255.0  # Normalize for visualization

def process_images(image_dir, mask_dir, output_dir, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    target_layer = model.model.backbone.layer4[-1]  # DeepLabV3 last ResNet layer
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    total, correct = 0, 0
    
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name)
        output_path = os.path.join(output_dir, image_name)
        
        if not os.path.exists(mask_path):
            continue
        
        image_tensor, image_np = load_image(image_path)
        image_tensor = image_tensor.to(device)
        
        output = model(image_tensor)
        prediction = torch.argmax(F.softmax(output, dim=1), dim=1).cpu().numpy()[0]
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (prediction.shape[1], prediction.shape[0]))
        
        correct += np.sum(prediction == mask)
        total += mask.size
        
        grayscale_cam = cam(input_tensor=image_tensor)[0, :]
        cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
        
        cv2.imwrite(output_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
        
    accuracy = correct / total if total > 0 else 0
    print(f"Segmentation Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    image_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\image_2"
    mask_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\semantic_rgb"
    output_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\Real-time-Semantic-Segmentation-Survey\Real-time-Semantic-Segmentation-Survey-main\output_pretrained_Future3_Grok\vit"
    model_path = r"C:\Users\Lenovo\Dropbox\PC\Desktop\Real-time-Semantic-Segmentation-Survey\Real-time-Semantic-Segmentation-Survey-main\model.pth"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Download model if missing
    download_model_if_missing(model_path)
    
    model = SegmentationModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    process_images(image_dir, mask_dir, output_dir, model)
