import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from transformers import ViTModel, ViTFeatureExtractor
import albumentations as A

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ViT model with segmentation head
def load_model():
    model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)
    model.segmentation_head = nn.Sequential(
        nn.Conv2d(768, 256, kernel_size=1),  # Reduce channels
        nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),  # Upsample to 224x224
        nn.Conv2d(256, 1, kernel_size=1)  # Output single channel
    ).to(device)
    model.eval()
    return model

# Load images and masks with resizing and augmentation
def load_images_and_masks(image_dir, mask_dir, target_size=(224, 224), augment=True):
    image_filenames = sorted(os.listdir(image_dir))
    mask_filenames = sorted(os.listdir(mask_dir))
    images = [cv2.imread(os.path.join(image_dir, f)) for f in image_filenames]
    masks = [cv2.imread(os.path.join(mask_dir, f), 0) for f in mask_filenames]
    
    # Resize to ViT input size
    resized_images = [cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR) for img in images]
    resized_masks = [cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST) for mask in masks]
    
    if augment:
        aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ])
        augmented_images, augmented_masks = [], []
        for img, mask in zip(resized_images, resized_masks):
            augmented = aug(image=img, mask=mask)
            augmented_images.append(augmented['image'])
            augmented_masks.append(augmented['mask'])
        resized_images.extend(augmented_images)
        resized_masks.extend(augmented_masks)
        image_filenames.extend([f"aug_{f}" for f in image_filenames])
    
    return resized_images, resized_masks, image_filenames, images, masks  # Return original images/masks too

# IoU calculation with label quality check
def calculate_iou(pred, target, evaluate_label_quality=False):
    pred = pred > 0.5
    target = target > 0
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    iou = intersection / union if union != 0 else 1.0
    
    if evaluate_label_quality:
        target_variance = np.var(target)
        if target_variance < 0.01:
            iou *= 0.9
    return iou

# Preprocessing for ViT
def preprocess_image(image):
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs['pixel_values'].to(device)  # Shape: [1, 3, 224, 224]

# Predict, evaluate, and save with resizing
def predict_and_save(model, images, masks, image_filenames, output_dir, original_images, original_masks):
    os.makedirs(output_dir, exist_ok=True)
    ious = []
    
    with torch.no_grad():
        for idx, (img, mask, orig_img, orig_mask, filename) in enumerate(zip(images, masks, original_images, original_masks, image_filenames)):
            input_tensor = preprocess_image(img)  # Shape: [1, 3, 224, 224]
            outputs = model(input_tensor).last_hidden_state  # Shape: [1, 197, 768]
            # Reshape patch embeddings (exclude CLS token)
            batch_size, num_patches, hidden_size = outputs.shape
            h = w = int((num_patches - 1) ** 0.5)  # 14x14 grid
            output = outputs[:, 1:, :].reshape(batch_size, h, w, hidden_size).permute(0, 3, 1, 2)  # [1, 768, 14, 14]
            pred = model.segmentation_head(output)  # Shape: [1, 1, 224, 224]
            pred = torch.sigmoid(pred).cpu().numpy()[0, 0]  # Shape: [224, 224]
            
            # Resize prediction to original mask size
            original_size = orig_mask.shape  # e.g., (375, 1242)
            pred_resized = cv2.resize(pred, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
            pred_mask = (pred_resized > 0.5).astype(np.uint8) * 255
            
            iou = calculate_iou(pred_mask, orig_mask, evaluate_label_quality=True)
            ious.append(iou)
            
            # Save results
            cv2.imwrite(os.path.join(output_dir, f"pred_{filename}"), pred_mask)
            cv2.imwrite(os.path.join(output_dir, f"gt_{filename}"), orig_mask)
            
            # XAI Visualization (using resized prediction)
            plt.imshow(pred_resized, cmap='hot')
            plt.title(f"Prediction Heatmap - IoU: {iou:.4f}")
            plt.savefig(os.path.join(output_dir, f"heatmap_{filename}.png"))
            plt.close()
    
    return ious

# Main execution
if __name__ == "__main__":
    image_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\image_2"
    mask_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\semantic_rgb"
    output_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\Real-time-Semantic-Segmentation-Survey\Real-time-Semantic-Segmentation-Survey-main\output_pretrained_Future3_Grok\vit"
    
    # Load data with resizing and augmentation
    images, masks, image_filenames, original_images, original_masks = load_images_and_masks(image_dir, mask_dir, target_size=(224, 224), augment=True)
    
    print("Evaluating model: ViT")
    model = load_model()
    os.makedirs(output_dir, exist_ok=True)
    ious = predict_and_save(model, images, masks, image_filenames, output_dir, original_images, original_masks)
    print(f"ViT: Mean IoU = {np.mean(ious):.4f}")