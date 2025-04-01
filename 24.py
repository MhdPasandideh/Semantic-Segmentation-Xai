import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models.segmentation as tv_models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader

# Define directories
INPUT_DIR = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\image_2"
LABEL_DIR = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\semantic_rgb"
OUTPUT_BASE = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\output_Xai_4models_Heatmaps"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)

# Custom Dataset class
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask.squeeze(0).long()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def preprocess_image(image_path, size=(256, 256)):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def preprocess_mask(mask_path, size=(256, 256)):
    mask = Image.open(mask_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    return transform(mask).squeeze().long()

# Compute IoU
def compute_iou(pred_mask, true_mask, num_classes=32):
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()
    iou_per_class = []
    for cls in range(num_classes):
        intersection = torch.logical_and(pred_mask == cls, true_mask == cls).sum().item()
        union = torch.logical_or(pred_mask == cls, true_mask == cls).sum().item()
        if union > 0:
            iou_per_class.append(intersection / union)
        else:
            iou_per_class.append(np.nan)
    return np.nanmean(iou_per_class)

# Grad-CAM for XAI
def generate_grad_cam(model, image_tensor, layer_name="backbone.layer4", device="cpu"):
    model.eval()
    image_tensor = image_tensor.to(device)
    
    activations = None
    def hook_fn(module, input, output):
        nonlocal activations
        activations = output
    
    layer_parts = layer_name.split('.')
    layer = model
    for part in layer_parts:
        layer = getattr(layer, part)
    hook = layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(image_tensor)
    
    hook.remove()
    
    heatmap = activations.squeeze().cpu().detach()
    heatmap = torch.mean(heatmap, dim=0)
    heatmap = F.relu(heatmap)
    heatmap = heatmap / (heatmap.max() + 1e-10)
    heatmap = heatmap.numpy()
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (256, 256))
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_colored, heatmap

# Evaluate model and save heatmap
def evaluate_and_save_heatmap(model, image_tensor, true_mask_tensor, img_file, output_dir, device="cpu", is_torchvision=False):
    model.eval()
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    true_mask_tensor = true_mask_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        pred = output['out'] if is_torchvision else output
        pred_mask = torch.argmax(pred, dim=1).squeeze().cpu()
    
    iou = compute_iou(pred_mask, true_mask_tensor.cpu())
    
    # Generate and save heatmap
    heatmap_colored, heatmap_raw = generate_grad_cam(model, image_tensor, device=device)
    heatmap_path = os.path.join(output_dir, f"heatmap_{img_file}.png")
    cv2.imwrite(heatmap_path, heatmap_colored)
    
    return iou, pred_mask, heatmap_colored

# Main execution
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
train_dataset = SegmentationDataset(INPUT_DIR, LABEL_DIR, transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define models
models_dict = {
    "DeepLabV3_ResNet50": (tv_models.deeplabv3_resnet50(pretrained=True), True),
    "DeepLabV3_ResNet101": (tv_models.deeplabv3_resnet101(pretrained=True), True),
    "FCN_ResNet50": (tv_models.fcn_resnet50(pretrained=True), True),
    "FCN_ResNet101": (tv_models.fcn_resnet101(pretrained=True), True),
}

# Process each model
for model_name, (model, is_torchvision) in models_dict.items():
    print(f"Evaluating {model_name}")
    model = model.to(device)
    
    output_dir = os.path.join(OUTPUT_BASE, model_name)
    os.makedirs(output_dir, exist_ok=True)
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    iou_scores = []
    for img_file in os.listdir(INPUT_DIR):
        image_path = os.path.join(INPUT_DIR, img_file)
        mask_path = os.path.join(LABEL_DIR, img_file)
        
        image_tensor = preprocess_image(image_path)
        true_mask_tensor = preprocess_mask(mask_path)
        
        iou, pred_mask, heatmap = evaluate_and_save_heatmap(
            model, image_tensor, true_mask_tensor, img_file, heatmap_dir, device, is_torchvision
        )
        iou_scores.append(iou)
        
        # Save predicted mask
        pred_mask_img = Image.fromarray(pred_mask.numpy().astype(np.uint8))
        pred_mask_img.save(os.path.join(output_dir, f"pred_{img_file}"))
        
        # Overlay heatmap on original image and save
        original_img = cv2.imread(image_path)
        original_img = cv2.resize(original_img, (256, 256))
        overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        cv2.imwrite(os.path.join(output_dir, f"overlay_{img_file}"), overlay)
    
    # Print IoU
    mean_iou = np.mean(iou_scores)
    print(f"{model_name} Mean IoU: {mean_iou:.4f}")
    
    # Plot IoU
    plt.figure()
    plt.bar(["Mean IoU"], [mean_iou], color='skyblue')
    plt.title(f"{model_name} Mean IoU")
    plt.ylabel("IoU Score")
    plt.savefig(os.path.join(output_dir, "iou_plot.png"))
    plt.close()

    # Example visualization for the first image
    img_file = os.listdir(INPUT_DIR)[0]
    image_tensor = preprocess_image(os.path.join(INPUT_DIR, img_file))
    true_mask_tensor = preprocess_mask(os.path.join(LABEL_DIR, img_file))
    iou, pred_mask, heatmap = evaluate_and_save_heatmap(
        model, image_tensor, true_mask_tensor, img_file, heatmap_dir, device, is_torchvision
    )
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(np.transpose(image_tensor.squeeze().cpu().numpy(), (1, 2, 0)))
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap='jet')
    plt.title(f"Predicted Mask (IoU: {iou:.4f})")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    
    plt.savefig(os.path.join(output_dir, "example_visualization.png"))
    plt.close()