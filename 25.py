import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import accuracy_score
from transformers import ViTForImageClassification

# **Paths**
image_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\image_2"
label_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\city scence dataset\KITTI\training\semantic_rgb"

# **Load Model**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device)
model.eval()

# **Preprocessing**
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# **Load Images & Labels**
image_files = sorted(os.listdir(image_dir))
label_files = sorted(os.listdir(label_dir))

accuracies = []  # Store accuracy values

for img_file, lbl_file in zip(image_files, label_files):
    img_path = os.path.join(image_dir, img_file)
    lbl_path = os.path.join(label_dir, lbl_file)

    # **Read and preprocess image**
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # **Read label**
    label = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)  # Ensure grayscale labels

    # **Predict**
    with torch.no_grad():
        outputs = model(image_tensor)
        pred = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]  # Ensure 2D output

    # **Ensure correct shape**
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)  # Make it 2D

    height, width = pred.shape
    label_resized = cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)

    # **Compute accuracy**
    acc = accuracy_score(label_resized.flatten(), pred.flatten())
    accuracies.append(acc)

# **Plot accuracy**
plt.figure(figsize=(10, 5))
plt.plot(accuracies, marker='o', linestyle='-')
plt.xlabel("Image Index")
plt.ylabel("Accuracy")
plt.title("Segmentation Accuracy Per Image")
plt.grid()
plt.savefig("segmentation_accuracy.png")  # Save accuracy plot
plt.show()

print(f"Average Accuracy: {np.mean(accuracies):.4f}")
