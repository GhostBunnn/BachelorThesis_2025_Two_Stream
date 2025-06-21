import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
from new_code.datasets.spatial_dataset_txt import RGBDataset
from models.spatial_model import load_spatial_model
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import json

# Hyperparameters
batch_size = 25
learning_rate = 5e-4
num_epochs = 10
early_stopping_patience = 15  # Patience for early stopping

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "extracted_rgb_frames")

# Dataset preparation
all_videos = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
dataset = RGBDataset(all_videos)  # Load dataset with all videos

# Splitting dataset into train, validation, and test sets
n = len(dataset)
n_train = int(n * 0.7)  # 70% for training
n_val = int(n * 0.15)   # 15% for validation
n_test = n - n_train - n_val  # Remaining 15% for testing

train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test])

print(f"Training videos: {len(train_dataset)}")
print(f"Validation videos: {len(val_dataset)}")
print(f"Testing videos: {len(test_dataset)}")

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

# Apply transformations to each split
train_dataset.dataset.transform = transform  # Apply transformations to the train dataset
val_dataset.dataset.transform = transform    # Apply to validation dataset
test_dataset.dataset.transform = transform   # Apply to test dataset

# Save class mapping
class_mapping_path = "class_mapping.json"
with open(class_mapping_path, "w") as f:
    json.dump(train_dataset.dataset.class_to_idx, f, indent=4)
print(f"Class mapping saved to {class_mapping_path}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
num_classes = len(train_dataset.dataset.class_to_idx)

# Adding Dropout to the Model
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

# Training setup
criterion = nn.CrossEntropyLoss()

# Optimizer Options
# Uncomment one of the following two lines to select an optimizer:
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,                  # Adjust based on batch size (try 0.1 for large batch sizes)
    # weight_decay=1e-4,        # Regularization to reduce overfitting
    # nesterov=True             # Nesterov momentum for smoother convergence
)

# Add a learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=1, verbose=True
)

# Early stopping variables
best_val_loss = float('inf')
patience_counter = 0

# Training loop
for epoch in range(num_epochs):
    # Training Phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Training Progress Bar
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

    # Validation Phase
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

    # Log Epoch Summary
    print(f"Epoch {epoch + 1}/{num_epochs}: "
          f"Train Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Step the scheduler
    scheduler.step(avg_val_loss)  # Use average validation loss

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# Testing Phase
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

# Save the model
torch.save({
    "model_state_dict": model.state_dict(),
    "class_to_idx": train_dataset.dataset.class_to_idx,
}, "spatial_stream_model.pth")
print("Final model and class mapping saved to spatial_stream_model.pth")
