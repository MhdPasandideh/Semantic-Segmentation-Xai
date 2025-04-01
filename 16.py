import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models.segmentation as models
import cv2
import json
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained segmentation model
def load_model(model_name):
    model_dict = {
        "deeplabv3_resnet50": models.deeplabv3_resnet50,
        "deeplabv3_resnet101": models.deeplabv3_resnet101,
        "fcn_resnet50": models.fcn_resnet50,
        "fcn_resnet101": models.fcn_resnet101,
        "vit_b_16": models.segmentation.deeplabv3_resnet50,  # Placeholder for ViT
    }
    
    if model_name not in model_dict:
        raise ValueError("Unsupported model name")
    
    model = model_dict[model_name](pretrained=True).to(device)
    model.eval()
    return model

# Load images and masks
def load_images_and_masks(image_dir, mask_dir):
    image_filenames = sorted(os.listdir(image_dir))
    mask_filenames = sorted(os.listdir(mask_dir))
    
    images = [cv2.imread(os.path.join(image_dir, f)) for f in image_filenames]
    masks = [cv2.imread(os.path.join(mask_dir, f), 0) for f in mask_filenames]
    
    # Ensure images and masks match
    if len(images) != len(masks):
        raise ValueError("Mismatch between images and masks count")
    
    return images, masks, image_filenames

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

# Main execution
if __name__ == "__main__":
    model_list = ["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101", "vit_b_16"]
    image_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\image_2"
    mask_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\semantic_rgb"
    output_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\Real-time-Semantic-Segmentation-Survey\Real-time-Semantic-Segmentation-Survey-main\ouput_pretrained_Future3_gpt"
    
    images, masks, image_filenames = load_images_and_masks(image_dir, mask_dir)
    iou_results = {}
    
    for model_name in model_list:
        print(f"Evaluating model: {model_name}")
        model = load_model(model_name)
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        ious = predict_and_save(model, images, masks, image_filenames, model_output_dir)
        mean_iou = np.mean(ious)
        print(f"{model_name}: Mean IoU = {mean_iou:.4f}")
        iou_results[model_name] = mean_iou
    
    # Save IoU results to file
    with open(os.path.join(output_dir, "iou_results.json"), "w") as f:
        json.dump(iou_results, f, indent=4)
    
    print("All models evaluated. IoU results saved.")
