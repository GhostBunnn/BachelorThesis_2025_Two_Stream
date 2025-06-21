import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from datasets.temporal_dataset_txt import TemporalDataset
from models.temporal_model import TemporalModel
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "temporal_model_lr0.0005_bs25_epochs25_03.pth")
DATA_DIR = os.path.join(BASE_DIR, "data", "extracted_optical_flow_frames")
TRAIN_SPLIT = os.path.join(BASE_DIR, "data", "splits", "trainlist03_processed.txt")
VAL_SPLIT = os.path.join(BASE_DIR, "data", "splits", "vallist03_processed.txt")
TEST_SPLIT = os.path.join(BASE_DIR, "data", "splits", "testlist03_processed.txt")
PRUNED_MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", "iterative_temporal_pruned_model.pth")

# Dataset transforms
transform = Compose([
    ToTensor(),
    Resize((224, 224), antialias=True),
    Normalize(mean=[0.5], std=[0.5])
])

# Dataset preparation
train_dataset = TemporalDataset(DATA_DIR, TRAIN_SPLIT, 10, transform=transform)
val_dataset = TemporalDataset(DATA_DIR, VAL_SPLIT, 10, transform=transform)
test_dataset = TemporalDataset(DATA_DIR, TEST_SPLIT, 10, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
num_classes = len(train_dataset.class_to_idx)

model = TemporalModel(num_classes).to(device)

def compute_channel_importance(layer):
    importance = torch.norm(layer.weight.data.view(layer.weight.size(0), -1), p=1, dim=1)
    return importance

def create_adaptive_mask(layer, prune_percentage):
    num_output_channels = layer.weight.size(0) 

    num_prune = int(prune_percentage * num_output_channels)

    importance = torch.norm(layer.weight.data.view(num_output_channels, -1), p=1, dim=1)

    _, indices = importance.sort() 
    prune_indices = indices[:num_prune] 

    mask = torch.ones(num_output_channels, dtype=torch.bool, device=layer.weight.device)
    mask[prune_indices] = False

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
    pruned_layer.weight.data = layer.weight.data[mask, :, :, :]
    if layer.bias is not None:
        pruned_layer.bias.data = layer.bias.data[mask]
    return pruned_layer


def replace_bn_with_pruned_version(bn, mask):
    pruned_bn = nn.BatchNorm2d(num_features=mask.sum().item())
    pruned_bn.weight.data = bn.weight.data[mask]
    pruned_bn.bias.data = bn.bias.data[mask]
    pruned_bn.running_mean = bn.running_mean[mask]
    pruned_bn.running_var = bn.running_var[mask]
    return pruned_bn


def prune_layer(layer, mask, next_bn=None):
    pruned_layer = replace_layer_with_pruned_version(layer, mask)

    if next_bn is not None:
        pruned_bn = replace_bn_with_pruned_version(next_bn, mask)
        return pruned_layer, pruned_bn

    return pruned_layer, None


def prune_channels(model, prune_percentage):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            mask = create_adaptive_mask(module, prune_percentage)

            # check if the next layer is a BatchNorm layer
            next_bn = None
            for next_name, next_module in model.named_modules():
                if name in next_name and next_name != name and isinstance(next_module, nn.BatchNorm2d):
                    next_bn = next_module
                    break
            pruned_layer, pruned_bn = prune_layer(module, mask, next_bn)
            module = pruned_layer
            if next_bn is not None:
                next_bn = pruned_bn

def retrain_model(model, train_loader, val_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Retrain Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

def test_model(model, test_loader):
    """ Evaluate the pruned model on the test dataset. """
    criterion = torch.nn.CrossEntropyLoss()
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

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return accuracy, avg_loss

# Plot results
def plot_metrics(prune_percentages, accuracies, baseline_accuracy, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(prune_percentages, accuracies, marker='o', label='Pruned Model Accuracy')
    plt.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline Accuracy')
    plt.title("Temporal Test Accuracy vs. Pruning Percentage")
    plt.xlabel("Pruning Percentage (%)")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    num_classes = len(test_dataset.class_to_idx)
    model = TemporalModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))

    # Evaluate baseline accuracy
    print("Evaluating baseline model...")
    baseline_accuracy, _ = test_model(model, test_loader)

    # Iterative pruning 
    max_prune_percentage = 0.2  # pruning limit
    prune_step = 0.08  # prune 8% at each step
    current_remaining_percentage = 1.0
    
    prune_percentages = []
    accuracies = []
    total_pruned_percentage = 0
    retrain_epochs = 10

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
        
        torch.save(model.state_dict(), PRUNED_MODEL_SAVE_PATH)
        print(f"Pruned model saved at: {PRUNED_MODEL_SAVE_PATH}")
        
        accuracy, _ = test_model(model, test_loader)
        prune_percentages.append(total_pruned_percentage * 100)
        accuracies.append(accuracy)


    print("Evaluating final pruned model on test data...")
    final_accuracy, final_loss = test_model(model, test_loader)

    plot_save_path = os.path.join(BASE_DIR, "plots", f"temporal_iterative_pruning{prune_step*100}_lr0.0001_retrain{retrain_epochs}_max20_03.png")
    plot_metrics(prune_percentages, accuracies, baseline_accuracy, plot_save_path)
    print(f"Plot saved to {plot_save_path}")
