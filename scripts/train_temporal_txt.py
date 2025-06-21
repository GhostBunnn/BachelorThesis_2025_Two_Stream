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

# Hyperparameters
batch_size = 64
learning_rate = 5e-4 #0.0001 #1e-3
num_epochs = 20
early_stopping_patience = 100

# Paths
FLOW_DIR = os.path.join(BASE_DIR, "data", "extracted_optical_flow_frames")
TRAIN_SPLIT = os.path.join(BASE_DIR, "data", "splits", "trainlist03_processed.txt")  # can be replaced with 01, 02, 03 depending on the txt file
VAL_SPLIT = os.path.join(BASE_DIR, "data", "splits", "vallist03_processed.txt")
TEST_SPLIT = os.path.join(BASE_DIR, "data", "splits", "testlist03_processed.txt")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", f"temporal_model_lr{learning_rate}_bs{batch_size}_epochs{num_epochs}.pth")

# Dataset transforms
transform = Compose([
    ToTensor(),
    Resize((224, 224), antialias=True),  # Resize to match model input
    Normalize(mean=[0.5], std=[0.5])  # Normalize flow frames
    # AddTransform()
])

# Dataset preparation
train_dataset = TemporalDataset(flow_dir=FLOW_DIR, split_file=TRAIN_SPLIT, num_frames=10, transform=transform)
val_dataset = TemporalDataset(flow_dir=FLOW_DIR, split_file=VAL_SPLIT, num_frames=10, transform=transform)
test_dataset = TemporalDataset(flow_dir=FLOW_DIR, split_file=TEST_SPLIT, num_frames=10, transform=transform)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# Save class mapping in the desired format
class_mapping_path = os.path.join(BASE_DIR, "scripts", "class_mapping_temporal.json")
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
num_classes = len(train_dataset.class_to_idx)  # Adjust this to the correct number of classes

model = TemporalModel(num_classes=num_classes).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Early stopping variables
best_val_loss = float('inf')
patience_counter = 0

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
            outputs = model(inputs)  # should be batchsize(64)*nr_labels(101)
            # print(f"Output shape from the model: {outputs.shape}")
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
        torch.save(model.state_dict(), "temporal_stream_best_model.pth")
        print(f"Model saved to {MODEL_SAVE_PATH}")
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# # Testing phase
# model.eval()
# test_loss = 0.0
# correct = 0
# total = 0

# print("Evaluating on test data...")
# with torch.no_grad():
#     with tqdm(test_loader, desc="Testing", unit="batch") as test_bar:
#         for inputs, labels in test_bar:
#             inputs, labels = inputs.to(device), labels.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             test_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             correct += (predicted == labels).sum().item()
#             total += labels.size(0)

#             test_bar.set_postfix(loss=f"{loss.item():.4f}")

# test_accuracy = 100 * correct / total
# avg_test_loss = test_loss / len(test_loader)

# print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
