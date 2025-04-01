import os
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor  # For ViT segmentation

# Define paths
image_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\image_2"
mask_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\semantic_rgb"
output_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\Real-time-Semantic-Segmentation-Survey\Real-time-Semantic-Segmentation-Survey-main\ouput_pretrained_Future3_gpt"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a segmentation model
def load_model(model_name):
    if model_name == "deeplabv3_resnet50":
        model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
    elif model_name == "deeplabv3_resnet101":
        model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device)
    elif model_name == "fcn_resnet50":
        model = models.segmentation.fcn_resnet50(pretrained=True).to(device)
    elif model_name == "fcn_resnet101":
        model = models.segmentation.fcn_resnet101(pretrained=True).to(device)
    elif model_name == "vit_b_16":  # ViT-based segmentation (Hugging Face)
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(device)
        model.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()  # Set model to evaluation mode
    return model

# Preprocess an image for segmentation
def preprocess_image(image_path, model_name):
    image = Image.open(image_path).convert("RGB")

    if model_name.startswith("vit"):  # ViT models require different preprocessing
        transform = load_model(model_name).feature_extractor
        inputs = transform(images=image, return_tensors="pt").to(device)
    else:
        transform = T.Compose([
            T.Resize((520, 520)),  # Resize for DeepLabV3 & FCN
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        inputs = transform(image).unsqueeze(0).to(device)

    return inputs

# Run and save segmentation results
def run_segmentation(model_name):
    model = load_model(model_name)
    image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        inputs = preprocess_image(image_path, model_name)

        with torch.no_grad():
            if model_name.startswith("vit"):
                output = model(**inputs).logits
            else:
                output = model(inputs)["out"]

        output_predictions = output.argmax(1).squeeze().cpu().numpy()  # Get class indices

        # Save segmentation result
        output_image = Image.fromarray((output_predictions * 255 / output_predictions.max()).astype("uint8"))
        output_path = os.path.join(output_dir, f"{model_name}_{image_file}")
        output_image.save(output_path)
        print(f"Saved: {output_path}")

# Run segmentation for all models
if __name__ == "__main__":
    model_list = ["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101", "vit_b_16"]

    for model_name in model_list:
        print(f"Processing {model_name}...")
        run_segmentation(model_name)

    print("Segmentation complete! Results saved in:", output_dir)
