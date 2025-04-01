import os
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50

# xplainable is not available, so we will create a simple heatmap instead.
# If you install xplainable, you can replace the simple heatmap with xplainable.

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

def create_simple_heatmap(output):
  """Create a simple heatmap from the model's output."""
  # This is a placeholder. You can replace this with a more sophisticated heatmap generation.
  return torch.mean(torch.abs(output["out"]), dim=1, keepdim=True)

if __name__ == "__main__":
    image_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\image_2"
    mask_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\semantic_rgb"
    output_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\Real-time-Semantic-Segmentation-Survey\Real-time-Semantic-Segmentation-Survey-main\ouput_pretrained_Future3_deepseek\xai"

    os.makedirs(output_dir, exist_ok=True)

    images, masks = load_images_and_masks(image_dir, mask_dir)

    model = deeplabv3_resnet50(pretrained=True)
    model.eval()

    predicted_masks = []
    heatmaps = []

    for i, image in enumerate(images):
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_mask = torch.argmax(output["out"], dim=1)
            predicted_mask_color = torch.zeros((1, 3, predicted_mask.shape[1], predicted_mask.shape[2]))

            color_map = {
                0: (0, 0, 0),
                1: (128, 64, 128),
                2: (0, 0, 142),
            }
            for class_label, color in color_map.items():
                mask_class = (predicted_mask == class_label)
                predicted_mask_color[:, 0, :, :][mask_class] = color[0]
                predicted_mask_color[:, 1, :, :][mask_class] = color[1]
                predicted_mask_color[:, 2, :, :][mask_class] = color[2]

            predicted_mask_color_pil = transforms.ToPILImage()(predicted_mask_color.squeeze().byte())
            predicted_masks.append(predicted_mask_color_pil)

        # Generate simple heatmap.
        heatmap = create_simple_heatmap(output)
        heatmaps.append(heatmap)

        heatmap_output_path = os.path.join(output_dir, f"heatmap_{i}.png")
        save_heatmap(heatmap, heatmap_output_path)

        predicted_mask_path = os.path.join(output_dir, f"predicted_mask_{i}.png")
        predicted_mask_color_pil.save(predicted_mask_path)

    accuracy = calculate_accuracy(predicted_masks, masks)
    print(f"Accuracy: {accuracy}")