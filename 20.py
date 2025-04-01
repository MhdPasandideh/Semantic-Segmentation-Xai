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
from transformers import ViTForImageClassification, ViTFeatureExtractor  # Vision Transformer
import albumentations as A  # For data augmentation

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained segmentation model with ViT option
def load_model(model_name, use_vit=False):
    if use_vit:
        # Use Vision Transformer for segmentation (simplified adaptation)
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
        # Modify ViT for segmentation (replace classification head with segmentation head)
        model.classifier = nn.Conv2d(768, 1, kernel_size=1)  # Simplified adaptation
    elif model_name == "deeplabv3_resnet50":
        model = models.deeplabv3_resnet50(pretrained=True).to(device)
    elif model_name == "deeplabv3_resnet101":
        model = models.deeplabv3_resnet101(pretrained=True).to(device)
    elif model_name == "fcn_resnet50":
        model = models.fcn_resnet50(pretrained=True).to(device)
    elif model_name == "fcn_resnet101":
        model = models.fcn_resnet101(pretrained=True).to(device)
    else:
        raise ValueError("Unsupported model name")
    model.eval()
    return model

# Load images and masks with data augmentation
def load_images_and_masks(image_dir, mask_dir, augment=True):
    image_filenames = sorted(os.listdir(image_dir))
    mask_filenames = sorted(os.listdir(mask_dir))
    images = [cv2.imread(os.path.join(image_dir, f)) for f in image_filenames]
    masks = [cv2.imread(os.path.join(mask_dir, f), 0) for f in mask_filenames]
    
    if augment:
        # Define augmentation pipeline (Survey 9)
        aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ])
        augmented_images, augmented_masks = [], []
        for img, mask in zip(images, masks):
            augmented = aug(image=img, mask=mask)
            augmented_images.append(augmented['image'])
            augmented_masks.append(augmented['mask'])
        images.extend(augmented_images)
        masks.extend(augmented_masks)
        image_filenames.extend([f"aug_{f}" for f in image_filenames])
    
    return images, masks, image_filenames

# IoU calculation with label quality check (Survey 11)
def calculate_iou(pred, target, evaluate_label_quality=False):
    pred = pred > 0.5
    target = target > 0
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    iou = intersection / union if union != 0 else 1.0
    
    if evaluate_label_quality:
        # Simple noise detection: Check for low variance in target (Survey 11)
        target_variance = np.var(target)
        if target_variance < 0.01:  # Threshold for noisy/poor label
            iou *= 0.9  # Penalize IoU for potentially noisy labels
    return iou

# Preprocessing with ViT compatibility
def preprocess_image(image, use_vit=False):
    if use_vit:
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        inputs = feature_extractor(images=image, return_tensors="pt")
        return inputs['pixel_values'].to(device)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0).to(device)

# Predict, evaluate, and save with XAI visualization (Survey 13)
def predict_and_save(model, images, masks, image_filenames, output_dir, use_vit=False):
    os.makedirs(output_dir, exist_ok=True)
    ious = []
    
    with torch.no_grad():
        for idx, (img, mask, filename) in enumerate(zip(images, masks, image_filenames)):
            input_tensor = preprocess_image(img, use_vit)
            output = model(input_tensor)['out'] if not use_vit else model(input_tensor).logits
            pred = torch.sigmoid(output).cpu().numpy()[0, 0] if not use_vit else output.squeeze().cpu().numpy()
            pred_mask = (pred > 0.5).astype(np.uint8) * 255
            
            iou = calculate_iou(pred_mask, mask, evaluate_label_quality=True)
            ious.append(iou)
            
            # Save results
            cv2.imwrite(os.path.join(output_dir, f"pred_{filename}"), pred_mask)
            cv2.imwrite(os.path.join(output_dir, f"gt_{filename}"), mask)
            
            # XAI Visualization (heatmap of prediction confidence)
            plt.imshow(pred, cmap='hot')
            plt.title(f"Prediction Heatmap - IoU: {iou:.4f}")
            plt.savefig(os.path.join(output_dir, f"heatmap_{filename}.png"))
            plt.close()
    
    return ious

# Main execution with external dataset integration
if __name__ == "__main__":
    model_list = ["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101", "vit"]
    image_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\image_2"
    mask_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\semantic_rgb"
    output_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\Real-time-Semantic-Segmentation-Survey\Real-time-Semantic-Segmentation-Survey-main\ouput_pretrained_Future3_Grok"
    
    # Load data with augmentation
    images, masks, image_filenames = load_images_and_masks(image_dir, mask_dir, augment=True)
    
    for model_name in model_list:
        print(f"Evaluating model: {model_name}")
        use_vit = (model_name == "vit")
        model = load_model(model_name, use_vit=use_vit)
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        ious = predict_and_save(model, images, masks, image_filenames, model_output_dir, use_vit=use_vit)
        print(f"{model_name}: Mean IoU = {np.mean(ious):.4f}")