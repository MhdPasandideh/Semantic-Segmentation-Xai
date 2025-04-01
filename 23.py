import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation

class SelfDrivingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert('RGB')
        # Convert RGB mask to class indices (assuming KITTI format)
        mask = Image.open(mask_path).convert('RGB')
        
        if self.transform:
            image = self.transform["image"](image)
            mask = self.transform["mask"](mask)
            
        # Convert RGB mask to class indices
        mask = rgb_to_class_indices(mask)
        
        return image, mask

def rgb_to_class_indices(mask):
    # Convert RGB mask to class indices (simplified for KITTI)
    # This is a basic implementation - adjust according to your specific class mapping
    mask = mask.numpy()
    mask = np.transpose(mask, (1, 2, 0))  # CHW to HWC
    class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    
    # Example mapping (adjust based on KITTI semantic_rgb classes)
    road = np.all(mask == [128, 64, 128], axis=2)
    class_mask[road] = 1
    
    return torch.from_numpy(class_mask)

def get_explainable_heatmap(model, image, device):
    model.eval()
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        logits = output.logits
        
        # Generate attention heatmap
        attention = torch.softmax(logits, dim=1)
        heatmap = attention.max(dim=1)[0].squeeze().cpu().numpy()
        
    return heatmap

def calculate_accuracy(pred_mask, true_mask):
    pred_mask_flat = pred_mask.flatten()
    true_mask_flat = true_mask.flatten()
    return accuracy_score(true_mask_flat, pred_mask_flat)

def save_heatmap(heatmap, original_image, output_path):
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(output_path, overlay)

def main():
    image_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\image_2"
    mask_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\semantic_rgb"
    output_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\Real-time-Semantic-Segmentation-Survey\Real-time-Semantic-Segmentation-Survey-main\output_pretrained_Future3_Grok\xai2"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate transforms for image and mask
    transform = {
        "image": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]),
        "mask": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    }
    
    dataset = SelfDrivingDataset(image_dir, mask_dir, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model with correct number of classes (adjust based on KITTI classes)
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/segformer-b0-finetuned-ade-512-512',
        num_labels=2  # Adjust based on your KITTI classes
    )
    model.to(device)
    
    total_accuracy = 0
    num_samples = 0
    
    for i, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        
        original_image = cv2.imread(os.path.join(image_dir, sorted(os.listdir(image_dir))[i]))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        heatmap = get_explainable_heatmap(model, images[0], device)
        
        with torch.no_grad():
            outputs = model(images)
            pred_masks = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        
        true_masks = masks.cpu().numpy()
        accuracy = calculate_accuracy(pred_masks[0], true_masks[0])
        total_accuracy += accuracy
        num_samples += 1
        
        output_path = os.path.join(output_dir, f'heatmap_{i:04d}.png')
        save_heatmap(heatmap, original_image, output_path)
        
        print(f"Image {i+1}/{len(dataset)} - Accuracy: {accuracy:.4f}")
    
    avg_accuracy = total_accuracy / num_samples
    print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
    
    with open(os.path.join(output_dir, 'accuracy_results.txt'), 'w') as f:
        f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
        f.write(f"Number of samples: {num_samples}\n")

if __name__ == "__main__":
    main()