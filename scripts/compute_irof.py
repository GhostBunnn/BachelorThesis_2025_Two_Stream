import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import matplotlib
from models.spatial_model import load_spatial_model
from models.temporal_model import TemporalModel
import argparse

'''
To run this code, us the commands:

#unpruned representations
python explanations.py --run_id run1

#pruned representations
# possible pruning values (up to 50%): 4.00, 7.84, 11.53, 15.07, 18.46, 21.72, 24.86, 
# 27.86, 30.75, 33.52, 36.18, 38.73, 41.18, 43.53, 45.79, 47.96, 50.04
python compute_irof.py --run_id run1 --use_pruned --prune_amount 4_00percent
'''

RGB_FRAME_ROOT = os.path.join(BASE_DIR, "data", "extracted_rgb_frames")
FLOW_FRAME_ROOT = os.path.join(BASE_DIR, "data", "extracted_optical_flow_frames")

script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "class_mapping.json"), "r") as f:
    class_mapping = json.load(f)
    
def load_temporal_model(num_classes, checkpoint_path, device):
    model = TemporalModel(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()

def load_spatial_model_with_dropout(num_classes, checkpoint_path, device):
    model = load_spatial_model(num_classes)
    # Safely get in_features whether fc is Linear or Sequential
    if isinstance(model.fc, nn.Sequential):
        for layer in model.fc:
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                break
        else:
            raise ValueError("No Linear layer found in model.fc Sequential.")
    else:
        in_features = model.fc.in_features

    # Replace with dropout + linear
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()

def find_file_with_prefix(directory, prefix):
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and (filename.endswith(".pth")):
            return os.path.join(directory, filename)
    raise FileNotFoundError(f"No file found starting with '{prefix}' in {directory}")

def load_spatial_input(path):
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def load_temporal_input(flow_folder, idx1=None, idx2=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    flow_tensors = []

    # Optical flow should have 10 x-flow and 10 y-flow => 20 channels total
    for i in range(10):
        x_path = os.path.join(flow_folder, f"flow_x_{i:04d}.jpg")
        y_path = os.path.join(flow_folder, f"flow_y_{i:04d}.jpg")

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise FileNotFoundError(f"Missing flow frames: {x_path} or {y_path}")

        x_img = Image.open(x_path).convert("L")
        y_img = Image.open(y_path).convert("L")

        x_tensor = transform(x_img)[0]  # shape [H, W]
        y_tensor = transform(y_img)[0]  # shape [H, W]

        flow_tensors.extend([x_tensor, y_tensor])

    input_tensor = torch.stack(flow_tensors, dim=0).unsqueeze(0)  # shape: [1, 20, 224, 224]
    return input_tensor

def load_saliency(path):
    saliency = Image.open(path).convert("L")
    saliency = np.array(saliency).astype(np.float32) / 255.0
    return saliency

def apply_mask(input_tensor, saliency, percent):
    k = int(percent * saliency.size)
    threshold = np.sort(saliency.ravel())[::-1][k-1]
    mask = (saliency >= threshold).astype(np.float32)

    # resize mask to match input
    if mask.shape != input_tensor.shape[-2:]:
        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()
        mask = torch.nn.functional.interpolate(mask, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        mask = mask.squeeze().numpy()

    masked_input = input_tensor.clone()
    for c in range(3):
        masked_input[0, c] *= torch.tensor(1 - mask).to(input_tensor.device)

    return masked_input

def compute_irof(model, input_tensor, saliency, target_class, steps=10, device='cpu'):
    input_tensor = input_tensor.to(device)
    model.eval()
    scores = []

    for p in range(0, 101, 100//steps):
        masked_input = apply_mask(input_tensor, saliency, percent=p / 100)
        with torch.no_grad():
            output = model(masked_input.to(device))
            prob = F.softmax(output, dim=1)[0, target_class].item()
        scores.append(prob)

    return scores

def overlay_mask_on_image(image, explanation, alpha=0.6, cmap="jet"):
    """
    Args:
        image (np.ndarray): (H, W, 3) RGB image, dtype=uint8
        explanation (np.ndarray): (H_orig, W_orig) saliency map, float32 in [0, 1]
    Returns:
        np.ndarray: overlay image (H, W, 3), dtype=uint8
    """
    # Resize explanation to match image
    if explanation.shape[:2] != image.shape[:2]:
        explanation = Image.fromarray((explanation * 255).astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]), resample=Image.BILINEAR
        )
        explanation = np.array(explanation).astype(np.float32) / 255.0

    explanation = np.clip(explanation, 0, 1)
    cmap_func = matplotlib.colormaps.get_cmap(cmap)
    heatmap = cmap_func(explanation)[:, :, :3]  # drop alpha
    heatmap = (heatmap * 255).astype(np.uint8)

    overlay = (1 - alpha) * image + alpha * heatmap
    return overlay.astype(np.uint8)


def process_stream(stream, model, saliency_root, overlay_dir, irof_dir, device):
    global_csv_path = os.path.join(irof_dir, f"all_{stream}_irof_scores.csv")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(irof_dir, exist_ok=True)

    with open(global_csv_path, "w") as global_csv:
        global_csv.write("video,frame_file,irof_auc\n")

    video_folders = sorted([v for v in os.listdir(saliency_root) if os.path.isdir(os.path.join(saliency_root, v))])
    
    for video_folder in tqdm(video_folders, desc=f"Processing {stream}"):
        video_name = video_folder.replace(f"{stream}_", "")
        video_path = os.path.join(saliency_root, video_folder)
        overlay_out = os.path.join(overlay_dir, video_folder)
        irof_out = os.path.join(irof_dir, video_folder)
        os.makedirs(overlay_out, exist_ok=True)
        os.makedirs(irof_out, exist_ok=True)

        if stream == "temporal":
            saliency_files = sorted([
                f for f in os.listdir(video_path)
                if f.startswith("saliency_") and f.endswith(".png")
            ])
        else:
            saliency_files = sorted([
                f for f in os.listdir(video_path)
                if f.startswith("saliency_frame") and f.endswith(".png")
            ])

        all_scores = []
        for saliency_file in saliency_files:
            if stream == "temporal":
                pair = saliency_file.replace("saliency_", "").replace(".png", "")
                try:
                    idx1, idx2 = map(int, pair.split("+"))
                except ValueError:
                    print(f"Invalid filename format: {saliency_file}")
                    continue
                frame_file = f"{pair}.jpg"
            else:
                frame_id = saliency_file.replace("saliency_frame", "").replace(".png", "")
                frame_file = f"frame_{int(frame_id):04d}.jpg"

            saliency_path = os.path.join(video_path, saliency_file)
            if not os.path.exists(saliency_path):
                continue

            saliency = load_saliency(saliency_path)

            if stream == "spatial":
                rgb_frame_path = os.path.join(RGB_FRAME_ROOT, video_name, frame_file.replace(".png", ".jpg"))
                input_tensor = load_spatial_input(rgb_frame_path)
                image_rgb = np.array(Image.open(rgb_frame_path).convert("RGB").resize((224, 224)))
            else:
                opt_frame_path = os.path.join(FLOW_FRAME_ROOT, video_name)
                input_tensor = load_temporal_input(opt_frame_path)
                image_rgb = np.stack([saliency * 255] * 3, axis=-1).astype(np.uint8)

            overlay_img = overlay_mask_on_image(image_rgb, saliency, alpha=0.6, cmap="jet")
            Image.fromarray(overlay_img).save(os.path.join(overlay_out, f"overlay_{frame_file}"))

            # Predict top class
            with torch.no_grad():
                output = model(input_tensor.to(device))
                top_class = output.argmax(dim=1).item()

            # IROF
            scores = compute_irof(model, input_tensor, saliency, top_class, device=device)
            x = np.linspace(0, 1, len(scores))
            irof_auc = auc(x, scores)

            all_scores.append(scores)

            # Log per-frame AUC
            with open(os.path.join(irof_out, "irof_scores.csv"), "a") as f:
                f.write(f"{frame_file},{irof_auc:.6f}\n")

        # Average curve and AUC per video
        if all_scores:
            mean_curve = np.mean(all_scores, axis=0)
            mean_auc = auc(x, mean_curve)

            # Plot only average IROF curve
            plt.figure()
            plt.plot(np.linspace(0, 100, len(mean_curve)), mean_curve, marker='o')
            plt.title(f"Average IROF – {video_folder}")
            plt.xlabel("% Top Salient Pixels Masked")
            plt.ylabel("Confidence")
            plt.grid(True)
            plt.savefig(os.path.join(irof_out, f"irof_average_{video_folder}.png"))
            plt.close()

            # with open(os.path.join(irof_out, "avg_irof_score.csv"), "w") as f:
            #     f.write(f"video,{video_folder}\nmean_auc,{mean_auc:.6f}\n")
            with open(global_csv_path, "a") as global_csv:
                global_csv.write(f"{video_folder},MEAN,{mean_auc:.6f}\n")
            tqdm.write(f"Finished {video_folder} (mean AUC: {mean_auc:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Two-stream IROF Scores")
    parser.add_argument('--use_pruned', action='store_true', help='Use pruned spatial and temporal models')
    parser.add_argument('--run_id', type=str, required=True, help='Model run id (e.g., run1)')
    parser.add_argument('--prune_amount', type=str, default=None,
                        help='Amount of pruning (e.g. 4_00percent) — required if using pruned models')
    args = parser.parse_args()
    if args.use_pruned and not args.prune_amount:
        parser.error("--prune_amount is required when using pruned models")
    model_variant = "pruned" if args.use_pruned else "unpruned"
    if args.use_pruned != True:
        args.prune_amount = 'unpruned'

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
    
    saliency_root = os.path.join(BASE_DIR, "plots", "saliency_maps")
    overlay_dir = os.path.join(BASE_DIR, "plots", "overlay_outputs")
    os.makedirs(overlay_dir, exist_ok=True)
    irof_dir = os.path.join(BASE_DIR, "plots", "irof_plots")
    os.makedirs(irof_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spatial_model = load_spatial_model_with_dropout(101, SPATIAL_MODEL_PATH, device)
    temporal_model = load_temporal_model(101, TEMPORAL_MODEL_PATH, device)

    stream_info = {
        "spatial": {
            "model": spatial_model,
            "saliency_root": os.path.join(saliency_root, args.run_id, args.prune_amount, "spatial"),
            "overlay_dir": os.path.join(overlay_dir, args.run_id, args.prune_amount, "spatial"),
            "irof_dir": os.path.join(irof_dir, args.run_id, args.prune_amount, "spatial"),
        },
        "temporal": {
            "model": temporal_model,
            "saliency_root": os.path.join(saliency_root, args.run_id, args.prune_amount, "temporal"),
            "overlay_dir": os.path.join(overlay_dir, args.run_id, args.prune_amount, "temporal"),
            "irof_dir": os.path.join(irof_dir, args.run_id, args.prune_amount, "temporal"),
        }
    }
    
    for stream, info in stream_info.items():
        process_stream(
            stream=stream,
            model=info["model"],
            saliency_root=info["saliency_root"],
            overlay_dir=info["overlay_dir"],
            irof_dir=info["irof_dir"],
            device=device
        )

if __name__ == "__main__":
    main()