import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from captum.attr import IntegratedGradients

# Verify versions
print(f"Versions - NumPy: {np.__version__}, Torch: {torch.__version__}")

def calculate_pixel_accuracy(predicted, true):
    """Calculates pixel-wise accuracy between predicted and true masks."""
    return np.mean(predicted == true)

def load_images_and_masks(image_dir, mask_dir):
    """Loads images and masks from specified directories."""
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png'))])
    
    if len(image_files) != len(mask_files):
        raise ValueError("Number of images and masks don't match")
    
    images = []
    masks = []
    
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        images.append(img)
        masks.append(mask)
    
    return images, masks

def preprocess_image(image):
    """Preprocesses the image for the model."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def rgb_to_labels(rgb_mask):
    """Converts RGB mask to label mask."""
    color_map = {
        (128, 64, 128): 1,  # Road
        (0, 0, 142): 2      # Car
    }
    label_mask = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
    for color, label in color_map.items():
        label_mask[(rgb_mask == color).all(axis=2)] = label
    return label_mask

def calculate_accuracy(predicted_masks, true_masks):
    """Calculates pixel-wise accuracy between predicted and true masks."""
    accuracies = []
    for pred_mask, true_mask in zip(predicted_masks, true_masks):
        pred_labels = rgb_to_labels(np.array(pred_mask))
        true_labels = rgb_to_labels(np.array(true_mask))
        accuracies.append(calculate_pixel_accuracy(pred_labels, true_labels))
    return np.mean(accuracies)

def save_heatmap(heatmap, output_path):
    """Saves the heatmap as an image."""
    heatmap_np = heatmap.squeeze().cpu().numpy()
    heatmap_norm = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min()) * 255
    heatmap_colored = cv2.applyColorMap(np.uint8(heatmap_norm), cv2.COLORMAP_JET)
    cv2.imwrite(output_path, heatmap_colored)

def generate_captum_heatmap(model, input_tensor):
    """Generates heatmap using Captum's Integrated Gradients."""
    ig = IntegratedGradients(model)
    attributions = ig.attribute(input_tensor, target=0)
    return torch.sum(torch.abs(attributions), dim=1, keepdim=True)

def apply_xai_correction(predicted_mask, heatmap, threshold=0.7):
    """Applies XAI-based correction to the predicted mask."""
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    uncertain_regions = heatmap_norm > threshold
    
    if uncertain_regions.any():
        certain_regions = ~uncertain_regions
        if certain_regions.any():
            mode_class = torch.mode(predicted_mask[certain_regions])[0]
            predicted_mask[uncertain_regions] = mode_class
    
    return predicted_mask

def calculate_xai_improvement(model, images, masks):
    """Calculates accuracy improvement after XAI corrections."""
    predicted_masks = []
    corrected_masks = []
    
    for image in images:
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_mask = torch.argmax(output["out"], dim=1).cpu()
            heatmap = generate_captum_heatmap(model, input_tensor)
            
            corrected_mask = apply_xai_correction(predicted_mask.clone(), heatmap)
            
            predicted_masks.append(transforms.ToPILImage()(predicted_mask.byte()))
            corrected_masks.append(transforms.ToPILImage()(corrected_mask.byte()))
    
    original_acc = calculate_accuracy(predicted_masks, masks)
    corrected_acc = calculate_accuracy(corrected_masks, masks)
    improvement = ((corrected_acc - original_acc) / original_acc) * 100
    
    return original_acc, corrected_acc, improvement

def benchmark_xai(model, images, masks, output_dir):
    """Benchmarks different XAI methods."""
    os.makedirs(os.path.join(output_dir, "benchmark"), exist_ok=True)
    
    for i, image in enumerate(images[:3]):  # Just benchmark first 3 for speed
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            
            # Simple heatmap
            simple_heatmap = torch.mean(torch.abs(output["out"]), dim=1, keepdim=True)
            save_heatmap(simple_heatmap, os.path.join(output_dir, "benchmark", f"simple_{i}.png"))
            
            # Captum heatmap
            captum_heatmap = generate_captum_heatmap(model, input_tensor)
            save_heatmap(captum_heatmap, os.path.join(output_dir, "benchmark", f"captum_{i}.png"))

def analyze_failure_modes(model, images, masks, output_dir):
    """Analyzes model failure cases."""
    os.makedirs(os.path.join(output_dir, "failures"), exist_ok=True)
    predicted_masks = []
    
    for i, (image, mask) in enumerate(zip(images, masks)):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_mask = torch.argmax(output["out"], dim=1).cpu()
            predicted_masks.append(transforms.ToPILImage()(predicted_mask.byte()))
            
            # Calculate accuracy for this sample
            pred_labels = rgb_to_labels(np.array(predicted_masks[-1]))
            true_labels = rgb_to_labels(np.array(mask))
            acc = calculate_pixel_accuracy(pred_labels, true_labels)
            
            if acc < 0.5:  # Save failure cases
                image.save(os.path.join(output_dir, "failures", f"image_{i}.png"))
                predicted_masks[-1].save(os.path.join(output_dir, "failures", f"pred_{i}.png"))
                mask.save(os.path.join(output_dir, "failures", f"true_{i}.png"))

def main():
    # Configuration
    # Define paths
    image_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\image_2"
    mask_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\semantic_rgb"
    output_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\Real-time-Semantic-Segmentation-Survey\Real-time-Semantic-Segmentation-Survey-main\ouput_pretrained_Future3_deepseek3"
    #image_dir = "/content/input3"
    #mask_dir = "/content/label3"
    #output_dir = "/content/output3"
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        images, masks = load_images_and_masks(image_dir, mask_dir)
        if not images:
            raise ValueError("No valid images found in the specified directories")
        
        # Load model
        model = deeplabv3_resnet50(pretrained=True)
        model.eval()
        
        # XAI analysis
        print("Running XAI analysis...")
        orig_acc, corr_acc, improvement = calculate_xai_improvement(model, images[:5], masks[:5])  # Use subset for demo
        print(f"Original Accuracy: {orig_acc:.4f}")
        print(f"Corrected Accuracy: {corr_acc:.4f}")
        print(f"Improvement: {improvement:.2f}%")
        
        # Benchmarking
        print("\nRunning XAI benchmark...")
        benchmark_xai(model, images[:3], masks[:3], output_dir)
        
        # Failure analysis
        print("\nAnalyzing failure modes...")
        analyze_failure_modes(model, images[:10], masks[:10], output_dir)
        
        print("\nAll operations completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()