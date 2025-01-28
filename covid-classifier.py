import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from PIL import Image
import argparse
import numpy as np
import random


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


TRAIN_DIR = os.path.join(BASE_DIR, 'covid_dataset', 'train')
EVAL_DIR = os.path.join(BASE_DIR, 'evaluation_Set')
RESULT_FILE = os.path.join(BASE_DIR, 'result.txt')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)



# Check if required directories exist
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
if not os.path.exists(EVAL_DIR):
    raise FileNotFoundError(f"Evaluation directory not found: {EVAL_DIR}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) 
])

# Load Training Dataset
train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)


labels = [label for _, label in train_data]

#stratified split
train_indices, val_indices = train_test_split(
    range(len(train_data)),
    test_size=0.2, 
    stratify=labels,  
    random_state=42  
)

# Subset datasets
train_dataset = torch.utils.data.Subset(train_data, train_indices)
val_dataset = torch.utils.data.Subset(train_data, val_indices)

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
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training Function
def train_model(epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

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


    unique, counts = np.unique(val_labels, return_counts=True)
    print("Class distribution in validation set:", dict(zip(unique, counts)))

    print("True labels:", val_labels)
    print("Predicted labels:", val_preds)

    # Calculate the basic F1 score
    val_f1 = f1_score(val_labels, val_preds)
    print(f"Validation F1 Score (basic): {val_f1 * 100:.4f}%")

    # Weighted F1 score
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

                img_name = img_names[0]  
                label = preds.item()
                f.write(f"{img_name:<20}{label}\n")

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