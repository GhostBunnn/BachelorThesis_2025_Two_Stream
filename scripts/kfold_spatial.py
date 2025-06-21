import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from models.spatial_model import load_spatial_model
from datasets.spatial_dataset_txt import RGBDataset

# Config
k = 5
epochs = 10
batch_size = 25
lr = 1e-4
total_pruned_percentage = 0.5  # total % of channels to prune
prune_steps = 4                # number of pruning rounds
finetune_epochs = 5

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data", "extracted_rgb_frames")
TRAIN_SPLIT = os.path.join(BASE_DIR, "data", "splits", "trainlist03_processed.txt")
VAL_SPLIT = os.path.join(BASE_DIR, "data", "splits", "vallist03_processed.txt")

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
train_dataset = RGBDataset(DATA_DIR, TRAIN_SPLIT, transform=transform)
val_dataset = RGBDataset(DATA_DIR, VAL_SPLIT, transform=transform)
full_dataset = ConcatDataset([train_dataset, val_dataset])
num_classes = len(train_dataset.class_to_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Functions ---

def train_model(model, train_loader, val_loader, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr, weight_decay=1e-4)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return evaluate_model(model, val_loader)

def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

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

def replace_layer_with_pruned_version(layer, out_mask, in_mask=None):
    in_channels = in_mask.sum().item() if in_mask is not None else layer.in_channels
    out_channels = out_mask.sum().item()

    pruned_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
        bias=layer.bias is not None
    )

    # Prune weight tensor
    w = layer.weight.data
    if in_mask is not None:
        w = w[:, in_mask]
    w = w[out_mask]
    pruned_layer.weight.data = w.clone()

    # Prune bias
    if layer.bias is not None:
        pruned_layer.bias.data = layer.bias.data[out_mask].clone()

    return pruned_layer

def prune_batchnorm_layer(layer, mask):
    pruned_bn = nn.BatchNorm2d(mask.sum().item())
    pruned_bn.weight.data = layer.weight.data[mask].clone()
    pruned_bn.bias.data = layer.bias.data[mask].clone()
    pruned_bn.running_mean = layer.running_mean[mask].clone()
    pruned_bn.running_var = layer.running_var[mask].clone()
    return pruned_bn

def iterative_prune_and_finetune(model, train_loader, val_loader, total_prune_pct, steps, finetune_epochs):
    remaining_pct = 1.0
    prune_step = 1 - (1 - total_prune_pct) ** (1 / steps)

    for step in range(steps):
        current_prune_pct = prune_step * remaining_pct
        remaining_pct *= (1 - prune_step)
        print(f"\n[Step {step + 1}/{steps}] Pruning {current_prune_pct*100:.2f}% of remaining channels...")

        modules_to_prune = []
        named_modules = list(model.named_modules())

        for i, (name, module) in enumerate(named_modules):
            if isinstance(module, nn.Conv2d):
                mask = create_adaptive_mask(module, current_prune_pct)
                new_conv = replace_layer_with_pruned_version(module, mask)
                modules_to_prune.append((name, new_conv))

                # Look ahead for matching BatchNorm2d
                for j in range(i + 1, len(named_modules)):
                    next_name, next_module = named_modules[j]
                    if isinstance(next_module, nn.BatchNorm2d) and next_module.num_features == mask.size(0):
                        new_bn = prune_batchnorm_layer(next_module, mask)
                        modules_to_prune.append((next_name, new_bn))
                        break

        for full_name, new_module in modules_to_prune:
            parent = model
            subnames = full_name.split(".")
            for sub in subnames[:-1]:
                parent = getattr(parent, sub)
            setattr(parent, subnames[-1], new_module)

        print(f"Retraining for {finetune_epochs} epochs after pruning...")
        train_model(model, train_loader, val_loader, finetune_epochs)

# --- Main K-Fold Loop ---
kf = KFold(n_splits=k, shuffle=True, random_state=42)
fold = 0
results_unpruned = []
results_pruned = []

for train_idx, val_idx in kf.split(full_dataset):
    print(f"\nFold {fold + 1}/{k}")

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size, shuffle=False, num_workers=4)

    # ----- UNPRUNED MODEL -----
    model_unpruned = load_spatial_model(num_classes, freeze_base=True).to(device)
    print(f"Training Unpruned")
    acc_unpruned = train_model(model_unpruned, train_loader, val_loader, epochs)
    results_unpruned.append(acc_unpruned)
    print(f"Unpruned Accuracy: {acc_unpruned:.2f}%")

    # Save trained weights before pruning
    unpruned_weights = model_unpruned.state_dict()

    # ----- ITERATIVELY PRUNED MODEL -----
    model_pruned = load_spatial_model(num_classes, freeze_base=True).to(device)
    model_pruned.load_state_dict(unpruned_weights)
    print(f"Training Pruned")
    iterative_prune_and_finetune(model_pruned, train_loader, val_loader,
                                 total_prune_pct=total_pruned_percentage,
                                 steps=prune_steps,
                                 finetune_epochs=finetune_epochs)
    acc_pruned = evaluate_model(model_pruned, val_loader)
    results_pruned.append(acc_pruned)
    print(f"Pruned Accuracy: {acc_pruned:.2f}%")

    fold += 1

# --- Bootstrap Confidence Intervals ---
def bootstrap_ci(data1, data2, n_bootstrap=1000, alpha=0.05):
    diffs = []
    combined = list(zip(data1, data2))
    for _ in range(n_bootstrap):
        sample = random.choices(combined, k=len(data1))
        diff = [b - a for a, b in sample]
        diffs.append(np.mean(diff))
    diffs.sort()
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    return lower, upper

mean_unpruned = np.mean(results_unpruned)
mean_pruned = np.mean(results_pruned)
ci_low, ci_high = bootstrap_ci(results_unpruned, results_pruned)

print("\n=== K-Fold Summary ===")
print(f"Unpruned: {mean_unpruned:.2f}% ± {np.std(results_unpruned):.2f}")
print(f"Pruned: {mean_pruned:.2f}% ± {np.std(results_pruned):.2f}")
print(f"Bootstrapped CI for (Pruned - Unpruned): [{ci_low:.2f}, {ci_high:.2f}]")

# Optional plot
plt.boxplot([results_unpruned, results_pruned], labels=["Unpruned", "Pruned"])
plt.title("K-Fold Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plots", "stats", "kfold_pruned_vs_unpruned.png"))