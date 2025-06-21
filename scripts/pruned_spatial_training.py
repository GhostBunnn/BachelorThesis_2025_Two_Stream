import sys
import os

# Dynamically add the base project directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.spatial_dataset_txt import RGBDataset
from models.spatial_model import load_spatial_model
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import prune
from tqdm import tqdm
import json

# Hyperparameters
batch_size = 25
learning_rate = 0.0001 #5e-4  # 1e-3
num_epochs = 20  # 30/40
early_stopping_patience = 100  # Patience for early stopping
prune_amount = 0.2  # Fraction of filters to prune (0 < prune_amount < 1)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Updated to project root
DATA_DIR = os.path.join(BASE_DIR, "data", "extracted_rgb_frames")
TRAIN_SPLIT = os.path.join(BASE_DIR, "data", "splits", "trainlist01_processed.txt")  # Change 01, 02, 03 as needed
VAL_SPLIT = os.path.join(BASE_DIR, "data", "splits", "vallist01_processed.txt")
TEST_SPLIT = os.path.join(BASE_DIR, "data", "splits", "testlist01_processed.txt")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", "best_spatial_model.pth")

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

# Save class mapping in the desired format
class_mapping_path = os.path.join(BASE_DIR, "scripts", "class_mapping.json")
with open(class_mapping_path, "w") as f:
    json.dump(train_dataset.class_to_idx, f, indent=4)
print(f"Class mapping saved to {class_mapping_path}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
num_classes = len(train_dataset.class_to_idx)

# Load model with dropout
def load_spatial_model_with_dropout(num_classes):
    """
    Load a ResNet-101 model with Dropout layers for regularization.
    """
    model = load_spatial_model(num_classes)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),  # Dropout before the final layer
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model

model = load_spatial_model_with_dropout(num_classes).to(device)

# Define channel importance computation
def compute_channel_importance(layer):
    importance = torch.norm(layer.weight.data.view(layer.weight.size(0), -1), p=1, dim=1)
    return importance

def create_adaptive_mask(layer, prune_percentage):
    """
    Create a mask for pruning the output channels of a convolutional layer.

    Args:
        layer (nn.Conv2d): The convolutional layer to be pruned.
        prune_percentage (float): The percentage of output channels to prune.

    Returns:
        torch.Tensor: A boolean mask for the output channels.
    """
    # Compute the number of output channels
    num_output_channels = layer.weight.size(0)  # Same as layer.out_channels

    # Compute the number of channels to prune
    num_prune = int(prune_percentage * num_output_channels)

    # Compute channel importance
    importance = torch.norm(layer.weight.data.view(num_output_channels, -1), p=1, dim=1)

    # Sort channels by importance and determine channels to prune
    _, indices = importance.sort()  # Ascending order of importance
    prune_indices = indices[:num_prune]  # Least important channels

    # Create a boolean mask for output channels
    mask = torch.ones(num_output_channels, dtype=torch.bool, device=layer.weight.device)
    mask[prune_indices] = False  # Mark channels to prune as False

    return mask

def replace_layer_with_pruned_version(layer, mask):
    """
    Replace a Conv2D layer with a pruned version.

    Args:
        layer (nn.Conv2d): The convolutional layer to replace.
        mask (torch.Tensor): Mask for the output channels.
    
    Returns:
        nn.Conv2d: A new convolutional layer with pruned output channels.
    """
    # Create a new Conv2D layer with pruned output channels
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
    # Copy pruned weights and biases
    pruned_layer.weight.data = layer.weight.data[mask, :, :, :]
    if layer.bias is not None:
        pruned_layer.bias.data = layer.bias.data[mask]
    return pruned_layer


def replace_bn_with_pruned_version(bn, mask):
    """
    Replace a BatchNorm2D layer with a pruned version.

    Args:
        bn (nn.BatchNorm2d): The BatchNorm layer to replace.
        mask (torch.Tensor): Mask for the pruned channels.
    
    Returns:
        nn.BatchNorm2d: A new BatchNorm layer with updated features.
    """
    # Create a new BatchNorm2D layer with pruned features
    pruned_bn = nn.BatchNorm2d(num_features=mask.sum().item())
    pruned_bn.weight.data = bn.weight.data[mask]
    pruned_bn.bias.data = bn.bias.data[mask]
    pruned_bn.running_mean = bn.running_mean[mask]
    pruned_bn.running_var = bn.running_var[mask]
    return pruned_bn


def prune_layer(layer, mask, next_bn=None):
    """
    Replace the convolutional layer and optionally the BatchNorm layer with pruned versions.

    Args:
        layer (nn.Conv2d): The convolutional layer to prune.
        mask (torch.Tensor): A boolean mask for output channels.
        next_bn (nn.BatchNorm2d, optional): The BatchNorm layer to update.
    """
    # Replace the convolutional layer
    pruned_layer = replace_layer_with_pruned_version(layer, mask)

    if next_bn is not None:
        # Replace the BatchNorm layer
        pruned_bn = replace_bn_with_pruned_version(next_bn, mask)
        return pruned_layer, pruned_bn

    return pruned_layer, None


def prune_channels(model, prune_percentage):
    """
    Prune channels from a model based on channel importance.

    Args:
        model (nn.Module): The model to be pruned.
        prune_percentage (float): Percentage of channels to prune.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Compute the mask for the current convolutional layer
            mask = create_adaptive_mask(module, prune_percentage)
            print(f"Pruning {name}: {module.weight.shape} -> Mask: {mask.shape}")

            # Check if the next layer is a BatchNorm layer
            next_bn = None
            for next_name, next_module in model.named_modules():
                if name in next_name and next_name != name and isinstance(next_module, nn.BatchNorm2d):
                    next_bn = next_module
                    break

            # Prune the layer and update associated BatchNorm, if applicable
            pruned_layer, pruned_bn = prune_layer(module, mask, next_bn)

            # Replace the original layers in the model
            module = pruned_layer
            if next_bn is not None:
                next_bn = pruned_bn

# Apply channel pruning
print("Applying channel pruning...")
prune_channels(model, prune_amount)
print("Channel pruning completed.")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

# Early stopping variables
best_val_loss = float('inf')
patience_counter = 0

# Prune the model
prune_channels(model, prune_amount)

# Training loop
for epoch in range(num_epochs):
    # Training phase
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

    # Validation phase
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
    avg_val_loss = val_loss / len(val_loader)

    # Log epoch summary
    print(f"Epoch {epoch + 1}/{num_epochs}: "
          f"Train Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Scheduler step
    scheduler.step(avg_val_loss)

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

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
