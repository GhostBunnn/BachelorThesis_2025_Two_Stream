import sys
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.spatial_dataset_txt import RGBDataset
from models.spatial_model import load_spatial_model
import torch.nn as nn
import torch.optim as optim
import csv
import argparse
from tqdm import tqdm

'''
run by typing python iterative_pruning_spatialplot.py --run_id run1, run2, run3...
'''
parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=str, required=True, help='Unique ID for this run (e.g., run1, run2, ...)')
args = parser.parse_args()
run_id = args.run_id

# Paths
# maybe best?
#MODEL_LOAD_PATH = os.path.join(BASE_DIR, "saved_models", "74.64%acc_sssspatial_model_lr0.0001_bs25_epochs25_03.pth")

#Adjust to same run unpruned model
MODEL_LOAD_PATH = os.path.join(BASE_DIR, "saved_models", "spatial", f"{run_id}_sssspatial_model_lr0.0001_bs25_epochs25_03.pth")

DATA_DIR = os.path.join(BASE_DIR, "data", "extracted_rgb_frames")
TRAIN_SPLIT = os.path.join(BASE_DIR, "data", "splits", "trainlist03_processed.txt")
VAL_SPLIT = os.path.join(BASE_DIR, "data", "splits", "vallist03_processed.txt")
TEST_SPLIT = os.path.join(BASE_DIR, "data", "splits", "testlist03_processed.txt")
PRUNED_MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", "spatial", f"{run_id}_iterative_spatial_pruned_model.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "spatial_pruning_accuracies.csv")

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

train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=25, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=25, shuffle=False, num_workers=4)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_dataset.class_to_idx)

model = load_spatial_model(num_classes, freeze_base=True).to(device)
model.load_state_dict(torch.load(MODEL_LOAD_PATH))

def compute_channel_importance(layer):
    return torch.norm(layer.weight.data.view(layer.weight.size(0), -1), p=1, dim=1)

def create_adaptive_mask(layer, prune_percentage):
    num_output_channels = layer.weight.size(0)
    num_prune = int(prune_percentage * num_output_channels)
    importance = compute_channel_importance(layer)
    _, indices = importance.sort()
    mask = torch.ones(num_output_channels, dtype=torch.bool, device=layer.weight.device)
    mask[indices[:num_prune]] = False
    return mask

def replace_layer_with_pruned_version(layer, mask):
    pruned_layer = nn.Conv2d(
        in_channels=layer.in_channels,
        out_channels=mask.sum().item(),
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
        bias=layer.bias is not None
    )
    pruned_layer.weight.data = layer.weight.data[mask]
    if layer.bias is not None:
        pruned_layer.bias.data = layer.bias.data[mask]
    return pruned_layer

def prune_channels(model, prune_percentage):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            mask = create_adaptive_mask(module, prune_percentage)
            module = replace_layer_with_pruned_version(module, mask)

def retrain_model(model, train_loader, val_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    l1_lambda = 1e-5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # L1 regularization
            l1_penalty = 0.0
            for name, param in model.named_parameters():
                if "bn" in name and "weight" in name:  # Target BatchNorm γ
                    l1_penalty += param.abs().sum()
            loss += l1_lambda * l1_penalty
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Retrain Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

def test_model(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
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
    return test_accuracy, avg_test_loss

def plot_metrics(prune_percentages, accuracies, losses, baseline_accuracy, baseline_loss, save_path):
    plt.figure(figsize=(10, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(prune_percentages, accuracies, marker='o', label='Pruned Model Accuracy')
    plt.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline Accuracy')
    plt.xlabel("Pruning Percentage (%)")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy vs. Spatial Pruning Percentage")
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(prune_percentages, losses, marker='o', label='Pruned Model Loss')
    plt.axhline(y=baseline_loss, color='r', linestyle='--', label='Baseline Loss')
    plt.xlabel("Pruning Percentage (%)")
    plt.ylabel("Loss")
    plt.title("Test Loss vs. Pruning Percentage")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Iterative pruning 
max_prune_percentage = 0.5  # pruning limit
prune_step = 0.04  # prune x% at each step
current_remaining_percentage = 1.0

print("Evaluating baseline model on test data...")
baseline_accuracy, baseline_loss = test_model(model, test_loader)  # baseline model

# Save baseline accuracy
with open(CSV_PATH, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([run_id, 0.0, baseline_accuracy, "baseline unpruned"])

prune_percentages = []
accuracies = []
losses = []
total_pruned_percentage = 0
retrain_epochs = 5

while total_pruned_percentage < max_prune_percentage:
    if total_pruned_percentage + (1 - current_remaining_percentage) > max_prune_percentage:
        break
    current_prune_percentage = prune_step * current_remaining_percentage
    print(f"Pruning {current_prune_percentage * 100:.2f}% of remaining channels...")
    prune_channels(model, current_prune_percentage)
    
    current_remaining_percentage *= (1 - prune_step)
    total_pruned_percentage = 1 - current_remaining_percentage
    
    print(f"Retraining after pruning {total_pruned_percentage * 100:.2f}%...")
    retrain_model(model, train_loader, val_loader, retrain_epochs)
    
    pruned_percent_str = f"{total_pruned_percentage * 100:.2f}".replace('.', '_')
    model_save_name = f"{run_id}_spatial_pruned_{pruned_percent_str}percent.pth"
    model_save_path = os.path.join(BASE_DIR, "saved_models", "spatial", model_save_name)

    torch.save(model.state_dict(), model_save_path)
    print(f"Pruned model saved at: {model_save_path}")
    
    accuracy, loss = test_model(model, test_loader)
    prune_percentages.append(total_pruned_percentage * 100)
    accuracies.append(accuracy)
    losses.append(loss)
    
    with open(CSV_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([run_id, total_pruned_percentage * 100, accuracy, "pruned"])

print("Evaluating final pruned model on test data...")
final_accuracy, _ = test_model(model, test_loader)

plot_save_path = os.path.join(BASE_DIR, "plots", "pruned_spatial", f"{args.run_id}_spatial_iterative_pruning{prune_step*100}_lr0.0001_retrain{retrain_epochs}_max50_03.png")
os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
plot_metrics(prune_percentages, accuracies, losses, baseline_accuracy, baseline_loss, plot_save_path)
print(f"Plot saved to {plot_save_path}")