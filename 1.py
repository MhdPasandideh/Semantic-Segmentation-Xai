import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import ResNet18_Weights  # For updated weights API

# Import models from the models directory
from models.DABNet import DABNet
from models.DSANet import DSANet
from models.EDANet import EDANet
from models.SPFNet import SPFNet
from models.SSFPN import SSFPN
from models.CGNet import CGNet
from models.ContextNet import ContextNet
from models.FastSCNN import FastSCNN

# Define paths
BASE_DIR = r"C:\Users\Lenovo\Dropbox\PC\Desktop\Real-time-Semantic-Segmentation-Survey\Real-time-Semantic-Segmentation-Survey-main"
INPUT_DIR = os.path.join(BASE_DIR, "contentn", "input")
LABEL_DIR = os.path.join(BASE_DIR, "content", "labels")
OUTPUT_DIR = os.path.join(BASE_DIR, "content output10")

# Create directories if they don't exist
for dir_path in [INPUT_DIR, LABEL_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Create subfolders for each model in output directory
model_names = ['DABNet', 'DSANet', 'CGNet', 'ContextNet', 'EDANet', 'FastSCNN', 'SPFNet', 'SSFPN']
for model_name in model_names:
    os.makedirs(os.path.join(OUTPUT_DIR, model_name), exist_ok=True)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((624, 1248)),  # Adjusted to match likely original size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to calculate IoU
def calculate_iou(pred, label):
    pred = pred.bool()
    label = label.bool()
    intersection = (pred & label).float().sum()
    union = (pred | label).float().sum()
    return intersection / (union + 1e-6)

# Modified build_model function
def build_model(model_name, num_classes):
    if model_name == 'DABNet':
        return DABNet(classes=num_classes)
    elif model_name == 'DSANet':
        return DSANet(classes=num_classes)
    elif model_name == 'CGNet':
        return CGNet(classes=num_classes)
    elif model_name == 'ContextNet':
        return ContextNet(classes=num_classes)
    elif model_name == 'EDANet':
        return EDANet(classes=num_classes)
    elif model_name == 'FastSCNN':
        return FastSCNN(classes=num_classes, aux=False)
    elif model_name == 'SPFNet':
        return SPFNet("resnet18", weights=ResNet18_Weights.IMAGENET1K_V1, classes=num_classes)  # Updated weights
    elif model_name == 'SSFPN':
        return SSFPN("resnet18", weights=ResNet18_Weights.IMAGENET1K_V1, classes=num_classes)  # Updated weights
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

# Main processing function
def process_and_evaluate(num_classes=19):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    models = {name: build_model(name, num_classes).to(device) for name in model_names}
    for model in models.values():
        model.eval()
    
    # Store IoU results
    iou_results = {name: [] for name in model_names}
    
    # Process each image
    image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg'))]
    if not image_files:
        raise FileNotFoundError(f"No .png or .jpg files found in {INPUT_DIR}. Please add input images.")
    
    for idx, image_file in enumerate(image_files):
        # Load image and label
        image_path = os.path.join(INPUT_DIR, image_file)
        label_path = os.path.join(LABEL_DIR, image_file)  # Assuming same filename
        
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        label = transforms.Resize((624, 1248), interpolation=Image.NEAREST)(label)  # Resize label
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        label_tensor = torch.from_numpy(np.array(label)).long().to(device)
        print(f"Processing {image_file} - Input tensor shape: {input_tensor.shape}")
        
        # Process each model
        for model_name, model in models.items():
            with torch.no_grad():
                output = model(input_tensor)
                if isinstance(output, tuple):  # Handle models with multiple outputs
                    output = output[0]
                
                # Get prediction
                pred = torch.argmax(output, dim=1).squeeze(0)
                
                # Calculate IoU
                iou = calculate_iou(pred, label_tensor)
                iou_results[model_name].append(iou.item())
                
                # Save prediction
                pred_img = Image.fromarray(pred.cpu().numpy().astype(np.uint8))
                output_path = os.path.join(OUTPUT_DIR, model_name, f"pred_{idx}.png")
                pred_img.save(output_path)
    
    return iou_results

# Plotting function
def plot_iou_results(iou_results):
    num_images = len(next(iter(iou_results.values())))
    
    # Plot IoU for each image
    plt.figure(figsize=(15, 10))
    for model_name, ious in iou_results.items():
        plt.plot(range(num_images), ious, label=model_name, marker='o')
    
    plt.xlabel('Image Index')
    plt.ylabel('IoU')
    plt.title('IoU Scores per Image for Different Models')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'iou_per_image.png'))
    plt.close()
    
    # Plot average IoU
    plt.figure(figsize=(10, 6))
    avg_ious = [np.mean(ious) for ious in iou_results.values()]
    plt.bar(iou_results.keys(), avg_ious)
    plt.xlabel('Model')
    plt.ylabel('Average IoU')
    plt.title('Average IoU Scores Across All Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'avg_iou.png'))
    plt.close()

if __name__ == "__main__":
    # Assuming 19 classes (common for Cityscapes-like datasets)
    num_classes = 19
    
    try:
        # Process images and get IoU results
        iou_results = process_and_evaluate(num_classes)
        
        # Plot results
        plot_iou_results(iou_results)
        
        # Print average IoU for each model
        print("\nAverage IoU Scores:")
        for model_name, ious in iou_results.items():
            print(f"{model_name}: {np.mean(ious):.4f}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")