import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from datasets.temporal_dataset_txt import TemporalDataset
from models.temporal_model import TemporalModel
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import argparse
import csv

# Argument parser for run_id
'''
run by typing python temporal_train_txt_plot.py --run_id run1, run2, run3... 
'''
parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=str, required=True, help='Unique ID for this run (e.g., run1, run2, ...)')
args = parser.parse_args()
run_id = args.run_id

# Hyperparameters
batch_size = 25
learning_rate = 5e-4 #1e-4
num_epochs = 25

# Paths
FLOW_DIR = os.path.join(BASE_DIR, "data", "extracted_optical_flow_frames")
TRAIN_SPLIT = os.path.join(BASE_DIR, "data", "splits", "trainlist03_processed.txt")
VAL_SPLIT = os.path.join(BASE_DIR, "data", "splits", "vallist03_processed.txt")
TEST_SPLIT = os.path.join(BASE_DIR, "data", "splits", "testlist03_processed.txt")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", "temporal", f"{args.run_id}_temporal_unpruned_lr{learning_rate}_bs{batch_size}_epochs{num_epochs}_03.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "temporal_pruning_accuracies.csv")
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Run ID", "Percentage Pruned", "Accuracy", "Model Type"])

# Dataset transforms
transform = Compose([
    ToTensor(),
    Resize((224, 224), antialias=True),
    Normalize(mean=[0.5], std=[0.5])
])

# Dataset preparation
train_dataset = TemporalDataset(flow_dir=FLOW_DIR, split_file=TRAIN_SPLIT, num_frames=10, transform=transform)
val_dataset = TemporalDataset(flow_dir=FLOW_DIR, split_file=VAL_SPLIT, num_frames=10, transform=transform)
test_dataset = TemporalDataset(flow_dir=FLOW_DIR, split_file=TEST_SPLIT, num_frames=10, transform=transform)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# Save class mapping
class_mapping_path = os.path.join(BASE_DIR, "scripts", "class_mapping_temporal.json")
with open(class_mapping_path, "w") as f:
    json.dump(train_dataset.class_to_idx, f, indent=4)
print(f"Class mapping saved to {class_mapping_path}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Running on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
num_classes = len(train_dataset.class_to_idx)
model = TemporalModel(num_classes=num_classes).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), weight_decay=1e-4, lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Early stopping variables
best_val_loss = float('inf')

# To store accuracy and loss for plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

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
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(train_accuracy)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
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
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Log epoch summary
    print(f"Epoch {epoch + 1}/{num_epochs}: "
          f"Train Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

# Testing phase
model.eval()
test_loss = 0.0
correct = 0
total = 0

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

print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

with open(CSV_PATH, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([run_id, 0.0, test_accuracy, "unpruned"])

plt.figure(figsize=(14, 6))  # Wider figure for side-by-side plots

# Subplot 1: Loss
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Subplot 2: Accuracy
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
combined_plot_path = os.path.join(BASE_DIR, "plots", "unpruned_temporal", f"{args.run_id}_temporal_combined_plot.png")
os.makedirs(os.path.dirname(combined_plot_path), exist_ok=True)
plt.savefig(combined_plot_path)
plt.show()

print(f"Combined plot saved to {combined_plot_path}")