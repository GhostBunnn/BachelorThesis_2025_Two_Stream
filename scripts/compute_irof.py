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

script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "class_mapping.json"), "r") as f:
    class_mapping = json.load(f)

def load_spatial_model_with_dropout(num_classes, checkpoint_path, device):
    model = load_spatial_model(num_classes)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_image(path):
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def load_image_rgb(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    return np.array(img)

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


def process_saliency_maps(saliency_root, model_path, overlay_dir, irof_dir, device):
    model = load_spatial_model_with_dropout(101, model_path, device)
    global_aurocs = []
    global_csv_path = os.path.join(irof_dir, "all_irof_scores.csv")
    with open(global_csv_path, "w") as global_csv:
        global_csv.write("video,frame_file,irof_auc\n")

    video_folders = sorted([v for v in os.listdir(saliency_root) if os.path.isdir(os.path.join(saliency_root, v))])
    for video_folder in tqdm(video_folders, desc="Processing videos", total=len(video_folders)):
        video_path = os.path.join(saliency_root, video_folder)
        if not os.path.isdir(video_path):
            continue

        overlay_out = os.path.join(overlay_dir, video_folder)
        irof_out = os.path.join(irof_dir, video_folder)
        os.makedirs(overlay_out, exist_ok=True)
        os.makedirs(irof_out, exist_ok=True)

        frame_files = sorted([f for f in os.listdir(video_path) if f.startswith("frame") and f.endswith(".png")])
        all_scores = []
        all_aurocs = []

        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            frame_id = os.path.splitext(frame_file)[0].replace("frame", "")
            saliency_path = os.path.join(video_path, f"saliency_frame{frame_id}.png")
            if not os.path.exists(saliency_path):
                continue

            image_rgb = load_image_rgb(frame_path)
            input_tensor = load_image(frame_path)
            saliency = load_saliency(saliency_path)

            # Overlay image (optional)
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
            all_aurocs.append(irof_auc)

            # Log per-frame AUC
            with open(os.path.join(irof_out, "irof_scores.csv"), "a") as f:
                f.write(f"{frame_file},{irof_auc:.6f}\n")
            # with open(global_csv_path, "a") as global_csv:
            #     global_csv.write(f"{video_folder},{frame_file},{irof_auc:.6f}\n")

        # Average curve and AUC per video
        if all_scores:
            mean_curve = np.mean(all_scores, axis=0)
            mean_auc = auc(x, mean_curve)
            global_aurocs.append(mean_auc)

            # Plot only average IROF curve
            plt.figure()
            plt.plot(np.linspace(0, 100, len(mean_curve)), mean_curve, marker='o')
            plt.title(f"Average IROF – {video_folder}")
            plt.xlabel("% Top Salient Pixels Masked")
            plt.ylabel("Confidence")
            plt.grid(True)
            plt.savefig(os.path.join(irof_out, f"irof_average_{video_folder}.png"))
            plt.close()

            with open(os.path.join(irof_out, "avg_irof_score.csv"), "w") as f:
                f.write(f"video,{video_folder}\nmean_auc,{mean_auc:.6f}\n")
            with open(global_csv_path, "a") as global_csv:
                global_csv.write(f"{video_folder},MEAN,{mean_auc:.6f}\n")
            tqdm.write(f"Finished {video_folder} (mean AUC: {mean_auc:.4f})")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Overlay saliency maps with Quantus and compute IROF.")
    parser.add_argument("--saliency_root", required=True, help="Root directory of saliency maps")
    parser.add_argument("--overlay_dir", default="saliency_overlay", help="Where to save overlay images")
    parser.add_argument("--irof_dir", default="irof_plots", help="Where to save IROF plots")
    
    model_path = os.path.join(BASE_DIR, "saved_models", "spatial_model_lr0.0001_bs25_epochs25_03.pth")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    process_saliency_maps(
        saliency_root=args.saliency_root,
        model_path=model_path,
        overlay_dir=args.overlay_dir,
        irof_dir=args.irof_dir,
        device=device
    )


if __name__ == "__main__":
    main()