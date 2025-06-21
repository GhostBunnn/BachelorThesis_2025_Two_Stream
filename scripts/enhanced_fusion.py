import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from datasets.spatial_dataset_txt import RGBDataset
from datasets.temporal_dataset_txt import TemporalDataset
from models.spatial_model import load_spatial_model
from models.temporal_model import TemporalModel
from itertools import zip_longest
import numpy as np
import torch.nn as nn
import joblib

# Paths
# unpruned
SPATIAL_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "spatial_model_lr0.0001_bs25_epochs25_03.pth")
TEMPORAL_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "temporal_model_lr0.0005_bs25_epochs25_03.pth")

# pruned
# SPATIAL_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "better2best_iterative_spatial_pruned_model.pth")
# TEMPORAL_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "Biterative_temporal_pruned_model.pth")

SPATIAL_DATA_DIR = os.path.join(BASE_DIR, "data", "extracted_rgb_frames")
TEMPORAL_DATA_DIR = os.path.join(BASE_DIR, "data", "extracted_optical_flow_frames")
TRAIN_SPLIT = os.path.join(BASE_DIR, "data", "splits", "trainlist03_processed.txt")
TEST_SPLIT = os.path.join(BASE_DIR, "data", "splits", "testlist03_processed.txt")

# Hyperparameters
batch_size = 32
SHOW_SOFTMAX = False
if SHOW_SOFTMAX:
    SVM_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "Softmax_fusion_svm.joblib")
else:
    SVM_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "representations_fusion_svm.joblib")

# Transformations
spatial_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

temporal_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Datasets and DataLoaders
spatial_dataset = RGBDataset(SPATIAL_DATA_DIR, TRAIN_SPLIT, transform=spatial_transform)
temporal_dataset = TemporalDataset(TEMPORAL_DATA_DIR, TRAIN_SPLIT, num_frames=10, transform=temporal_transform)

spatial_loader = DataLoader(spatial_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
temporal_loader = DataLoader(temporal_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_spatial_model_with_dropout(num_classes):
    model = load_spatial_model(num_classes)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model

# Spatial Stream
spatial_model = load_spatial_model_with_dropout(len(spatial_dataset.class_to_idx))
spatial_model.load_state_dict(torch.load(SPATIAL_MODEL_PATH))
spatial_model = spatial_model.to(device)
spatial_model.eval()

# Temporal Stream
temporal_model = TemporalModel(num_classes=len(temporal_dataset.class_to_idx))
temporal_model.load_state_dict(torch.load(TEMPORAL_MODEL_PATH))
temporal_model = temporal_model.to(device)
temporal_model.eval()

training = False
# Load or Train SVM
if os.path.exists(SVM_MODEL_PATH):
    print("Loading pre-trained SVM model...")
    fusion_svm = joblib.load(SVM_MODEL_PATH)
else:
    print("Training a new SVM model...")
    training = True
    features = []
    labels = []

    # Iterate over spatial and temporal loaders
    with torch.no_grad():
        for batch_pair in zip(spatial_loader, temporal_loader):
            spatial_batch = batch_pair[0]
            temporal_batch = batch_pair[1]

            spatial_inputs, spatial_labels = spatial_batch
            temporal_inputs, temporal_labels = temporal_batch

            # put both inputs on the same device
            spatial_inputs, spatial_labels = spatial_inputs.to(device), spatial_labels.to(device)
            temporal_inputs = temporal_inputs.to(device)

            # extract features
            spatial_features = spatial_model(spatial_inputs).view(spatial_inputs.size(0), -1).cpu().numpy()
            temporal_features = temporal_model(temporal_inputs).cpu().numpy()

            # Ensure batch sizes match
            min_batch_size = min(spatial_features.shape[0], temporal_features.shape[0])
            spatial_features = spatial_features[:min_batch_size]
            temporal_features = temporal_features[:min_batch_size]
            
            if SHOW_SOFTMAX:
                # Row-wise softmax
                max_row_spatial = np.max(spatial_features, axis=1, keepdims=True)
                e_x_spatial = np.exp(spatial_features - max_row_spatial)
                sum_row_spatial = np.sum(e_x_spatial, axis=1, keepdims=True)
                spatial_features = e_x_spatial / sum_row_spatial

                max_row_temp = np.max(temporal_features, axis=1, keepdims=True)
                e_x_temp = np.exp(temporal_features - max_row_temp)
                sum_row_temp = np.sum(e_x_temp, axis=1, keepdims=True)
                temporal_features = e_x_temp / sum_row_temp

            # Concatenate features
            combined_features = np.hstack((spatial_features, temporal_features))

            # Append features and labels
            features.append(combined_features)
            labels.append(spatial_labels[:min_batch_size].cpu().numpy())

    features = np.vstack(features)
    labels = np.concatenate(labels)

    # Train SVM
    param_grid = {
        'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 0.0001, 5e-4, 0.2]
    }
    
    svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
    grid_search = GridSearchCV(
        estimator=svm_pipeline,
        param_grid=param_grid,
        scoring='accuracy',
        cv=10,
        n_jobs=-1,
        verbose=3
    )

    grid_search.fit(features, labels)
    fusion_svm = grid_search.best_estimator_

    print("Best Params:", grid_search.best_params_)
    print("Best Cross-Val Score:", grid_search.best_score_)

    # Save SVM model
    joblib.dump(fusion_svm, SVM_MODEL_PATH)
    print(f"SVM model saved to {SVM_MODEL_PATH}")

# Testing
print("Evaluating fusion SVM model...")
correct = 0
total = 0

spatial_dataset = RGBDataset(SPATIAL_DATA_DIR, TEST_SPLIT, transform=spatial_transform)
temporal_dataset = TemporalDataset(TEMPORAL_DATA_DIR, TEST_SPLIT, num_frames=10, transform=temporal_transform)

spatial_test_loader = DataLoader(spatial_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
temporal_test_loader = DataLoader(temporal_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

with torch.no_grad():
    for batch_pair in zip(spatial_test_loader, temporal_test_loader):
        spatial_batch = batch_pair[0]
        temporal_batch = batch_pair[1]
        
        if spatial_batch is None or temporal_batch is None:
            continue

        spatial_inputs, spatial_labels = spatial_batch
        temporal_inputs, temporal_labels = temporal_batch
        
        spatial_inputs = spatial_inputs.to(device)
        temporal_inputs = temporal_inputs.to(device)
        
        # extract features
        spatial_features = spatial_model(spatial_inputs).view(spatial_inputs.size(0), -1).cpu().numpy()
        temporal_features = temporal_model(temporal_inputs).cpu().numpy()

        if SHOW_SOFTMAX:
            max_row_spatial = np.max(spatial_features, axis=1, keepdims=True)
            e_x_spatial = np.exp(spatial_features - max_row_spatial)
            sum_row_spatial = np.sum(e_x_spatial, axis=1, keepdims=True)
            spatial_features = e_x_spatial / sum_row_spatial

            max_row_temp = np.max(temporal_features, axis=1, keepdims=True)
            e_x_temp = np.exp(temporal_features - max_row_temp)
            sum_row_temp = np.sum(e_x_temp, axis=1, keepdims=True)
            temporal_features = e_x_temp / sum_row_temp
        
        min_batch_size = min(spatial_features.shape[0], temporal_features.shape[0])
        spatial_features = spatial_features[:min_batch_size]
        temporal_features = temporal_features[:min_batch_size]
        combined_features = np.hstack((spatial_features, temporal_features))

        # Predict
        predictions = fusion_svm.predict(combined_features)
        spatial_labels_np = spatial_labels[:min_batch_size].cpu().numpy()
        correct += np.sum(predictions == spatial_labels_np)
        total += spatial_labels.size(0)

test_accuracy = 100 * correct / total
print(f"Fusion SVM Test Accuracy: {test_accuracy:.2f}%")

if training: # include accuracy in name
    accuracy_str = f"{test_accuracy:.2f}"
    new_filename = SVM_MODEL_PATH.replace(".joblib", f"_acc_{accuracy_str}.joblib")
    
    if os.path.exists(SVM_MODEL_PATH):
        os.rename(SVM_MODEL_PATH, new_filename)
        print(f"Renamed model file to: {new_filename}")
    else:
        print("Warning: initial model file not found, cannot rename.")