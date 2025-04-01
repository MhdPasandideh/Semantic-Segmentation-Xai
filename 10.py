import os
import cv2
import torch
import numpy as np
import torchvision.models.segmentation as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import jaccard_score
from torchvision.transforms import functional as TF
import random

def get_pretrained_model(model_name, num_classes):
    if model_name == 'deeplabv3_resnet50':
        model = models.deeplabv3_resnet50(pretrained=True)
    elif model_name == 'deeplabv3_resnet101':
        model = models.deeplabv3_resnet101(pretrained=True)
    elif model_name == 'fcn_resnet50':
        model = models.fcn_resnet50(pretrained=True)
    elif model_name == 'fcn_resnet101':
        model = models.fcn_resnet101(pretrained=True)
    else:
        raise ValueError("Model not supported")
    
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)  # Adjust output classes
    return model

class SelfDrivingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 256))
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 256), interpolation=cv2.INTER_NEAREST)
        
        if self.transform:
            image = self.transform(image)
        
        mask = torch.from_numpy(mask).long()
        return image, mask

def augment_data(image, mask):
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
    return image, mask

def evaluate_models(model_list, image_dir, mask_dir, output_dir, num_classes=2, batch_size=4, augmentation=False, external_data=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SelfDrivingDataset(image_dir, mask_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    iou_scores = {}
    
    for model_name in model_list:
        model = get_pretrained_model(model_name, num_classes).to(device)
        model.eval()
        
        model_output_dir = os.path.join(output_dir, model_name)
        if augmentation:
            model_output_dir += "_augmented"
        if external_data:
            model_output_dir += "_external_data"

        os.makedirs(model_output_dir, exist_ok=True)
        pred_output_dir = os.path.join(model_output_dir, "predictions")
        os.makedirs(pred_output_dir, exist_ok=True)
        
        iou_list = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                if augmentation:
                    augmented_inputs, augmented_labels = [], []
                    for j in range(inputs.size(0)):
                        aug_input, aug_label = augment_data(inputs[j], labels[j])
                        augmented_inputs.append(aug_input)
                        augmented_labels.append(aug_label)
                    inputs = torch.stack(augmented_inputs).to(device)
                    labels = torch.stack(augmented_labels).to(device)

                outputs = model(inputs)['out']
                predicted = torch.argmax(outputs, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                
                for j in range(inputs.size(0)):
                    pred_img = predicted[j] * 255
                    cv2.imwrite(os.path.join(pred_output_dir, f"{i*batch_size+j}.png"), pred_img)
                    
                    iou = jaccard_score(labels[j].flatten(), predicted[j].flatten(), average='macro')
                    iou_list.append(iou)
        
        mean_iou = np.mean(iou_list)
        iou_scores[model_name] = mean_iou
        print(f"Model: {model_name}, Mean IoU: {mean_iou:.4f}")
    
    plt.figure()
    plt.bar(iou_scores.keys(), iou_scores.values())
    plt.xlabel("Model")
    plt.ylabel("Mean IoU")
    plt.title("IoU Comparison Across Models")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, "iou_comparison.png"))
    plt.close()

if __name__ == "__main__":
    model_list = ["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101"]
    image_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\image_2"
    mask_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\semantic_rgb"
    output_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\Real-time-Semantic-Segmentation-Survey\Real-time-Semantic-Segmentation-Survey-main\ouput_pretrained_Future3_deepseek"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Baseline
    evaluate_models(model_list, image_dir, mask_dir, output_dir)
    
    # Data Augmentation
    evaluate_models(model_list, image_dir, mask_dir, output_dir, augmentation=True)

    # Note: Implementing other research directions (External data, ViT, XAI, etc.) requires more complex code and potentially external libraries.
    # For example, ViT requires replacing the model architecture, External data requires a different dataset, and XAI requires specific interpretation techniques.