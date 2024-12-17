import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
from PIL import Image
import argparse
import numpy as np

# Base directory of the project (where the script resides)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct paths relative to BASE_DIR
TRAIN_DIR = os.path.join(BASE_DIR, 'covid_dataset', 'train')
EVAL_DIR = os.path.join(BASE_DIR, 'evaluation_Set')
RESULT_FILE = os.path.join(BASE_DIR, 'result.txt')

# Check if required directories exist
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
if not os.path.exists(EVAL_DIR):
    raise FileNotFoundError(f"Evaluation directory not found: {EVAL_DIR}")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] range
])

# Load Training Dataset
train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)

# Split into Train and Validation sets
train_size = int(0.9 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load Evaluation Dataset
def load_eval_images(eval_dir):
    return [os.path.join(eval_dir, fname) for fname in os.listdir(eval_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

eval_image_paths = load_eval_images(EVAL_DIR)

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, os.path.basename(img_path)

eval_dataset = EvalDataset(eval_image_paths, transform)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

# Model: ResNet18
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Modify output layer for binary classification
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Function
def train_model(epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Validation Function
def validate_model():
    model.eval()
    val_labels = []
    val_preds = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

    # Check for class distribution in validation set
    unique, counts = np.unique(val_labels, return_counts=True)
    print("Class distribution in validation set:", dict(zip(unique, counts)))

    print("True labels:", val_labels)
    print("Predicted labels:", val_preds)

    # Calculate the basic F1 score
    val_f1 = f1_score(val_labels, val_preds)
    print(f"Validation F1 Score (basic): {val_f1 * 100:.4f}%")

    # Weighted F1 score (if dataset is imbalanced)
    val_f1_weighted = f1_score(val_labels, val_preds, average='weighted')
    print(f"Validation Weighted F1 Score: {val_f1_weighted:.4f}")

    return val_f1

# Evaluation Function
def evaluate_model():
    model.eval()
    with open(RESULT_FILE, 'w') as f:
        with torch.no_grad():
            for inputs, img_names in eval_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # Write results to file
                img_name = img_names[0]  # Batch size is 1
                label = preds.item()
                f.write(f"{img_name} {label}\n")

    print(f"Evaluation completed. Results saved to {RESULT_FILE}")

# Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COVID-19 X-Ray Classifier")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()

    print("Starting training...")
    train_model(args.epochs)
    print("Training completed.")

    print("Validating model...")
    validate_model()

    print("Evaluating model on unseen data...")
    evaluate_model()