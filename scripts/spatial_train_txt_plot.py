import sys
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)  # Add the project root (new_code) to Python's search path

from datasets.spatial_dataset_txt import RGBDataset
from models.spatial_model import load_spatial_model
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import argparse
import csv

# Argument parser for run_id
'''
run by typing python spatial_train_txt_plot.py --run_id run1, run2, run3... 
'''
parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=str, required=True, help='Unique ID for this run (e.g., run1, run2, ...)')
args = parser.parse_args()
run_id = args.run_id

# Hyperparameters
batch_size = 25
learning_rate = 0.0001
num_epochs = 25

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data", "extracted_rgb_frames")
TRAIN_SPLIT = os.path.join(BASE_DIR, "data", "splits", "trainlist03_processed.txt")
VAL_SPLIT = os.path.join(BASE_DIR, "data", "splits", "vallist03_processed.txt")
TEST_SPLIT = os.path.join(BASE_DIR, "data", "splits", "testlist03_processed.txt")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", "spatial", f"{args.run_id}_spatial_unpruned_lr{learning_rate}_bs{batch_size}_epochs{num_epochs}_03.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "spatial_pruning_accuracies.csv")
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Run ID", "Percentage Pruned", "Accuracy", "Model Type"])

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset preparation
train_dataset = RGBDataset(DATA_DIR, TRAIN_SPLIT, transform=transform)
val_dataset = RGBDataset(DATA_DIR, VAL_SPLIT, transform=transform)
test_dataset = RGBDataset(DATA_DIR, TEST_SPLIT, transform=transform)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# Save class mapping
class_mapping_path = os.path.join(BASE_DIR, "scripts", "class_mapping.json")
with open(class_mapping_path, "w") as f:
    json.dump(train_dataset.class_to_idx, f, indent=4)
print(f"Class mapping saved to {class_mapping_path}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

num_classes = len(train_dataset.class_to_idx)
model = load_spatial_model(num_classes, freeze_base=True).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), weight_decay=1e-4, lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

# Metrics tracking
train_accuracies, val_accuracies = [], []
train_losses, val_losses = [], []

# Early stopping variables
best_val_loss = float('inf')

# Training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Training)", unit="batch") as train_bar:
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

    train_accuracy = 100 * correct / total
    train_accuracies.append(train_accuracy)
    train_losses.append(running_loss / len(train_loader))

    # Validation phase
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Validation)", unit="batch") as val_bar:
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                val_bar.set_postfix(loss=f"{loss.item():.4f}")

    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch + 1}/{num_epochs}: "
          f"Train Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%")

    # Scheduler step
    scheduler.step(val_loss / len(val_loader))

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

# Testing phase
model.eval()
test_loss, correct, total = 0.0, 0, 0

print("Evaluating on test data...")
with torch.no_grad():
    with tqdm(test_loader, desc="Testing", unit="batch") as test_bar:
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            test_bar.set_postfix(loss=f"{loss.item():.4f}")

test_accuracy = 100 * correct / total
avg_test_loss = test_loss / len(test_loader)

print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

with open(CSV_PATH, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([run_id, 0.0, test_accuracy, "unpruned"])

plt.figure(figsize=(14, 6))

# Loss plots
# Subplot 1: Loss
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Accuracy plots
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

# Add test accuracy at the bottom
plt.suptitle(f"Test Accuracy: {test_accuracy:.2f}%", fontsize=14, y=0.95)

# Save and display the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the test accuracy text
combined_plot_path = os.path.join(BASE_DIR, "plots", "unpruned_spatial", f"{args.run_id}_spatial_combined_plot.png")
os.makedirs(os.path.dirname(combined_plot_path), exist_ok=True)
plt.savefig(combined_plot_path)
plt.show()
plt.savefig(os.path.join(BASE_DIR, "spatial_combined_plot_lr5e4.png"))