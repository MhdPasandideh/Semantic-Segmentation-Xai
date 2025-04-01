import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models.segmentation as models
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Augmentation
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# Load pretrained segmentation model
def load_model(model_name):
    if model_name == "deeplabv3_resnet50":
        model = models.deeplabv3_resnet50(pretrained=True).to(device)
    elif model_name == "deeplabv3_resnet101":
        model = models.deeplabv3_resnet101(pretrained=True).to(device)
    elif model_name == "fcn_resnet50":
        model = models.fcn_resnet50(pretrained=True).to(device)
    elif model_name == "fcn_resnet101":
        model = models.fcn_resnet101(pretrained=True).to(device)
    elif model_name == "vit":
        model = sam_model_registry["vit_b"](checkpoint="path_to_vit_checkpoint").to(device)
    else:
        raise ValueError("Unsupported model name")
    model.eval()
    return model

# IoU calculation
def calculate_iou(pred, target):
    pred = pred > 0.5
    target = target > 0
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / union if union != 0 else 1.0

# Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Predict and evaluate
def predict_and_save(model, images, masks, image_filenames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ious = []
    
    with torch.no_grad():
        for idx, (img, mask, filename) in enumerate(zip(images, masks, image_filenames)):
            input_tensor = preprocess_image(img)
            output = model(input_tensor)['out']
            pred = torch.sigmoid(output).cpu().numpy()[0, 0]
            pred_mask = (pred > 0.5).astype(np.uint8) * 255
            
            iou = calculate_iou(pred_mask, mask)
            ious.append(iou)
            
            # Save results
            cv2.imwrite(os.path.join(output_dir, f"pred_{filename}"), pred_mask)
            cv2.imwrite(os.path.join(output_dir, f"gt_{filename}"), mask)
    
    return ious

# Explainable AI (XAI) - Grad-CAM
def apply_grad_cam(model, image_tensor):
    model.eval()
    image_tensor.requires_grad_()
    output = model(image_tensor)['out']
    output[:, 1].backward()
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activations(image_tensor).detach()
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    return heatmap

# Main execution
if __name__ == "__main__":
    model_list = ["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101", "vit"]
    image_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\image_2"
    mask_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\semantic_rgb"
    output_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\Real-time-Semantic-Segmentation-Survey\Real-time-Semantic-Segmentation-Survey-main\ouput_pretrained_Future3_gemini"
    
    # Data augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CustomDataset(image_dir, mask_dir, transform=transform)
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    images, masks, image_filenames = load_images_and_masks(image_dir, mask_dir)
    
    for model_name in model_list:
        print(f"Evaluating model: {model_name}")
        model = load_model(model_name)
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        ious = predict_and_save(model, images, masks, image_filenames, model_output_dir)
        print(f"{model_name}: Mean IoU = {np.mean(ious):.4f}")
        
        # Apply Grad-CAM for explainability
        sample_image, _ = dataset[0]
        sample_image = sample_image.unsqueeze(0).to(device)
        heatmap = apply_grad_cam(model, sample_image)
        plt.imshow(heatmap.cpu(), cmap='hot')
        plt.savefig(os.path.join(model_output_dir, "heatmap.png"))