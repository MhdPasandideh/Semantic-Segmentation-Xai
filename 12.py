import os
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from captum.attr import IntegratedGradients  # Using captum instead of xplainable.

def load_images_and_masks(image_dir, mask_dir):
    """Loads images and masks from specified directories."""
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

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

def calculate_accuracy(predicted_masks, true_masks):
    """Calculates the pixel-wise accuracy between predicted and true masks."""
    accuracies = []
    for pred_mask, true_mask in zip(predicted_masks, true_masks):
        pred_mask_np = np.array(pred_mask)
        true_mask_np = np.array(true_mask)

        pred_labels = rgb_to_labels(pred_mask_np)
        true_labels = rgb_to_labels(true_mask_np)

        acc = accuracy_score(true_labels.flatten(), pred_labels.flatten())
        accuracies.append(acc)

    return np.mean(accuracies)

def rgb_to_labels(rgb_mask):
    """Simple RGB to label conversion. Replace with your actual color mapping."""
    road_color = (128, 64, 128)
    car_color = (0, 0, 142)
    label_mask = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)

    label_mask[(rgb_mask == road_color).all(axis=2)] = 1  # Road
    label_mask[(rgb_mask == car_color).all(axis=2)] = 2  # Car

    return label_mask

def save_heatmap(heatmap, output_path):
    """Saves the heatmap as an image."""
    heatmap_np = heatmap.squeeze().cpu().numpy()
    heatmap_normalized = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min()) * 255
    heatmap_colored = cv2.applyColorMap(np.uint8(heatmap_normalized), cv2.COLORMAP_JET)
    cv2.imwrite(output_path, heatmap_colored)

def generate_captum_heatmap(model, input_tensor):
    """Generates heatmap using Captum's Integrated Gradients."""
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(input_tensor, target=0, return_convergence_delta=True)
    attributions = torch.sum(torch.abs(attributions), dim=1, keepdim=True)  # Sum absolute values along the channel dimension.
    return attributions

def benchmark_xai(model, images, masks, output_dir):
    """Benchmarks XAI by comparing different heatmap generation methods."""
    os.makedirs(os.path.join(output_dir, "benchmark"), exist_ok=True)
    heatmaps_simple = []
    heatmaps_captum = []

    for i, image in enumerate(images):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)

        heatmaps_simple.append(torch.mean(torch.abs(output["out"]), dim=1, keepdim=True))
        heatmaps_captum.append(generate_captum_heatmap(model, input_tensor))

        heatmap_simple_path = os.path.join(output_dir, "benchmark", f"simple_heatmap_{i}.png")
        save_heatmap(heatmaps_simple[-1], heatmap_simple_path)

        heatmap_captum_path = os.path.join(output_dir, "benchmark", f"captum_heatmap_{i}.png")
        save_heatmap(heatmaps_captum[-1], heatmap_captum_path)

    print("Benchmark complete")

def analyze_failure_modes(model, images, masks, output_dir):
    """Analyzes failure modes by identifying areas of low accuracy."""
    os.makedirs(os.path.join(output_dir, "failure_modes"), exist_ok=True)
    predicted_masks = []
    for i, image in enumerate(images):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_mask = torch.argmax(output["out"], dim=1).cpu()
            predicted_masks.append(transforms.ToPILImage()(predicted_mask.byte()))

    accuracy = calculate_accuracy(predicted_masks, masks)
    print(f"Average Accuracy: {accuracy}")
    for i in range(len(images)):
        if accuracy < 0.5:
            images[i].save(os.path.join(output_dir, "failure_modes", f"failure_image_{i}.png"))
            predicted_masks[i].save(os.path.join(output_dir, "failure_modes", f"failed_predicted_mask_{i}.png"))
    print("Failure modes analysis complete")

if __name__ == "__main__":
    image_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\image_2"
    mask_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\semantic_rgb"
    output_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\Real-time-Semantic-Segmentation-Survey\Real-time-Semantic-Segmentation-Survey-main\output_pretrained_Future3_deepseek\xai"

    os.makedirs(output_dir, exist_ok=True)

    images, masks = load_images_and_masks(image_dir, mask_dir)

    model = deeplabv3_resnet50(pretrained=True)
    model.eval()

    benchmark_xai(model, images, masks, output_dir)
    analyze_failure_modes(model, images, masks, output_dir)