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
from datasets.spatial_dataset_txt import RGBDataset
from datasets.temporal_dataset_txt import TemporalDataset
from models.spatial_model import load_spatial_model
from models.temporal_model import TemporalModel
import numpy as np
import torch.nn as nn
import argparse
import joblib
import csv


def find_file_with_prefix(directory, prefix):
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and (filename.endswith(".pth") or filename.endswith(".joblib")):
            return os.path.join(directory, filename)
    raise FileNotFoundError(f"No file found starting with '{prefix}' in {directory}")

'''
Running commands:

#unpruned representations
python fusion.py --run_id run1

#unpruned softmax
python fusion.py --run_id run1 --use_softmax

#possible pruned amounts: 4.00, 7.84, 11.53, 15.07, 18.64, 21.72, 24.86, 27.86
#in order to properly recognize the models, the emoung should be: 4_00percent, 11_53percent, etc

#pruned representations
python fusion.py --run_id run1 --use_pruned --prune_amount 4_00percent

#pruned softmax
python fusion.py --run_id run2 --use_pruned --prune_amount 7_84percent --use_softmax
'''
parser = argparse.ArgumentParser(description="Two-stream SVM fusion")
parser.add_argument('--use_softmax', action='store_true', help='Apply softmax to the features before fusion')
parser.add_argument('--use_pruned', action='store_true', help='Use pruned spatial and temporal models')
parser.add_argument('--run_id', type=str, required=True, help='Model run id (e.g., run1)')
parser.add_argument('--prune_amount', type=str, default=None,
                    help='Amount of pruning (e.g., 4_00) — required if using pruned models')
args = parser.parse_args()
if args.use_pruned and not args.prune_amount:
    parser.error("--prune_amount is required when using pruned models")
SHOW_SOFTMAX = args.use_softmax
fusion_type = "softmax" if SHOW_SOFTMAX else "representations"
model_variant = "pruned" if args.use_pruned else "unpruned"

# Paths
if args.use_pruned: # pruned
    SPATIAL_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "spatial", f"{args.run_id}_spatial_pruned_{args.prune_amount}.pth")
    TEMPORAL_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "temporal", f"{args.run_id}_temporal_pruned_{args.prune_amount}.pth")
else: # unpruned
    model_path = os.path.join(BASE_DIR, "saved_models", "spatial")
    prefix = f"{args.run_id}_spatial_{model_variant}"
    SPATIAL_MODEL_PATH = find_file_with_prefix(model_path, prefix)
    model_path = os.path.join(BASE_DIR, "saved_models", "temporal")
    prefix = f"{args.run_id}_temporal_{model_variant}"
    TEMPORAL_MODEL_PATH = find_file_with_prefix(model_path, prefix)

SPATIAL_DATA_DIR = os.path.join(BASE_DIR, "data", "extracted_rgb_frames")
TEMPORAL_DATA_DIR = os.path.join(BASE_DIR, "data", "extracted_optical_flow_frames")
TRAIN_SPLIT = os.path.join(BASE_DIR, "data", "splits", "trainlist03_processed.txt")
TEST_SPLIT = os.path.join(BASE_DIR, "data", "splits", "testlist03_processed.txt")

# Hyperparameters
batch_size = 32
prune_str = f"_pruned_{args.prune_amount}" if args.use_pruned else "unpruned"
SVM_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "fusion", f"{fusion_type}", f"{args.run_id}_{prune_str}_{fusion_type}_fusion_svm.joblib")

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

# Spatial Stream
spatial_model = load_spatial_model(len(spatial_dataset.class_to_idx))
spatial_model.load_state_dict(torch.load(SPATIAL_MODEL_PATH))
spatial_model = spatial_model.to(device)
spatial_model.eval()

# Temporal Stream
temporal_model = TemporalModel(num_classes=len(temporal_dataset.class_to_idx))
temporal_model.load_state_dict(torch.load(TEMPORAL_MODEL_PATH))
temporal_model = temporal_model.to(device)
temporal_model.eval()

svm_path = os.path.join(BASE_DIR, "saved_models", "fusion", f"{fusion_type}")
prefix = f"{args.run_id}_{prune_str}_{fusion_type}"

# Load or Train SVM
try:
    existing_model_path = find_file_with_prefix(svm_path, prefix)
    print("Loading pre-trained SVM model...")
    fusion_svm = joblib.load(existing_model_path)
    training = False
except FileNotFoundError:
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
    fusion_svm = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
    fusion_svm.fit(features, labels)

    # Save SVM model
    os.makedirs(os.path.dirname(SVM_MODEL_PATH), exist_ok=True)
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
        
# Logging to CSV
# will add the accuracy even if the fused model's accuracy is already saved
csv_path = os.path.join(BASE_DIR, "results", "fusion_results_" + f"{fusion_type}.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

log_data = {
    "run_id": args.run_id,
    "fusion_type": fusion_type,
    "pruned": "pruned" if args.use_pruned else "unpruned",
    "prune_amount": args.prune_amount if args.use_pruned else "N/A",
    "test_accuracy": round(test_accuracy, 2),
    "svm_model_path": new_filename if training else existing_model_path
}

write_header = not os.path.exists(csv_path)

with open(csv_path, mode='a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=log_data.keys())
    if write_header:
        writer.writeheader()
    writer.writerow(log_data)

print(f"Logged results to: {csv_path}")
