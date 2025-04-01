import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.DABNet import DABNet
from models.DSANet import DSANet
from models.EDANet import EDANet
from models.SPFNet import SPFNet
from models.SSFPN import SSFPN
from models.CGNet import CGNet
from models.ContextNet import ContextNet
from models.FastSCNN import FastSCNN

def build_model(model_name, num_classes):
    if model_name == 'DABNet':
        return DABNet(classes=num_classes)
    elif model_name == 'DSANet':
        return DSANet(classes=num_classes)
    elif model_name == 'CGNet':
        return CGNet(classes=num_classes)
    elif model_name == 'ContextNet':
        return ContextNet(classes=num_classes)
    elif model_name == 'EDANet':
        return EDANet(classes=num_classes)
    elif model_name == 'FastSCNN':
        return FastSCNN(classes=num_classes, aux=False)
    elif model_name == 'SPFNet':
        return SPFNet("resnet18", pretrained=True, classes=num_classes)
    elif model_name == 'SSFPN':
        return SSFPN("resnet18", pretrained=True, classes=num_classes)
    else:
        raise NotImplementedError

class RADDataset(Dataset):
    def __init__(self, input_dir, label_dir, transform=None, label_mapping=None):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.input_images = sorted(os.listdir(input_dir))
        self.label_images = sorted(os.listdir(label_dir))
        self.transform = transform
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        label_path = os.path.join(self.label_dir, self.label_images[idx])

        input_image = cv2.imread(input_path)
        label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (256, 256))
        label_image = cv2.resize(label_image, (256, 256), interpolation=cv2.INTER_NEAREST)

        if self.label_mapping:
            label_image_remapped = np.zeros_like(label_image)
            for old_label, new_label in self.label_mapping.items():
                label_image_remapped[label_image == old_label] = new_label
            label_image = label_image_remapped

        if self.transform:
            input_image = self.transform(input_image)
            label_image = torch.from_numpy(label_image).long()

        return input_image, label_image

def train_and_evaluate(model_name, input_dir, label_dir, output_dir, num_classes=2, batch_size=4, epochs=10, learning_rate=1e-3, label_mapping=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = RADDataset(input_dir, label_dir, transform=transform, label_mapping=label_mapping)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = build_model(model_name, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            labels = labels.squeeze(1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()

            for j in range(inputs.size(0)):
                output_path = os.path.join(output_dir, f"{i*batch_size+j}.png")
                cv2.imwrite(output_path, predicted[j] * 255)

if __name__ == "__main__":
    model_name = "CGNet"  # Choose your model
    input_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\image_2"
    label_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\semantic_rgb"
    output_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\Real-time-Semantic-Segmentation-Survey\Real-time-Semantic-Segmentation-Survey-main\output"

    num_classes = 2 # Change this based on your actual number of classes

    label_mapping = {
        119: 1, #example: if 119 must be class 1. Change it according to your data.
        # Add other mappings as needed.
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_and_evaluate(model_name, input_dir, label_dir, output_dir, num_classes=num_classes, label_mapping=label_mapping)

