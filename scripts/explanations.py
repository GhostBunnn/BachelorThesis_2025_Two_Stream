import os
import sys
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from models.spatial_model import load_spatial_model
from models.temporal_model import TemporalModel
import torch.nn as nn
from PIL import Image
import argparse
import csv

def find_file_with_prefix(directory, prefix):
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and (filename.endswith(".pth")):
            return os.path.join(directory, filename)
    raise FileNotFoundError(f"No file found starting with '{prefix}' in {directory}")

def load_temporal_model(num_classes, checkpoint_path, device):
    model = TemporalModel(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_spatial_model_dropout(num_classes, checkpoint_path, device):
    model = load_spatial_model(num_classes)

    # Check if model.fc is a Sequential already
    if isinstance(model.fc, nn.Sequential):
        # Safely extract from the second layer (assuming it's Linear)
        in_features = model.fc[1].in_features
    else:
        in_features = model.fc.in_features

    # Replace with new dropout + classifier
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def unnormalize_spatial(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(-1, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(-1, 1, 1)
    return tensor * std + mean

def unnormalize_temporal(tensor, mean=0.5, std=0.5):
    return tensor * std + mean


def vanilla_saliency(model, input_tensor, target_class=None):
    model.eval()
    model.zero_grad()

    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1)

    selected_logit = output[range(output.shape[0]), target_class].sum()
    selected_logit.backward()
    return input_tensor.grad.abs()


def save_saliency_map(saliency, path):
    plt.figure(figsize=(5, 5))
    plt.axis('off')  # No axes
    plt.imshow(saliency, cmap='hot', interpolation='nearest')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove padding
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_image(tensor_img, path, cmap=None):
    if isinstance(tensor_img, np.ndarray):
        img_np = tensor_img
    else:
        img = tensor_img.detach().cpu()
        if cmap:
            img_np = img.numpy()
        else:
            img_np = img.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)

    plt.imsave(path, img_np, cmap=cmap)

def process_video(video_folder, stream, model, transform, data_dir, device, global_saliency_storage=None):
    video_path = os.path.join(data_dir, video_folder)
    out_dir = os.path.join(BASE_DIR, "plots", "saliency_maps", f"{args.run_id}", f"{args.prune_amount}", f"{stream}", f"{stream}_{video_folder}")
    os.makedirs(out_dir, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])
    print(f"{stream.capitalize()} stream has {len(frame_files)} frames for video '{video_folder}'")

    if stream == 'spatial':
        saliency_accumulator = []

        frames = []
        for frame_name in frame_files:
            img = Image.open(os.path.join(video_path, frame_name)).convert("RGB")
            frames.append(transform(img))

        for i, t_img in enumerate(frames):
            t_img = t_img.unsqueeze(0).to(device).to(torch.float32)
            t_img.requires_grad_()

            saliency = vanilla_saliency(model, t_img, target_class=torch.tensor([0], device=device))
            sal_np = saliency[0].detach().cpu().numpy().mean(axis=0)
            saliency_accumulator.append(sal_np)

            # save_saliency_map(sal_np, f"Spatial Saliency frame#{i}", os.path.join(out_dir, f"saliency_frame{i}.png"))
            save_saliency_map(sal_np, os.path.join(out_dir, f"saliency_frame{i}.png"))
            # save_saliency_map(sal_np, os.path.join(out_dir, f"saliency_frame{i}.png"))
            np.save(os.path.join(out_dir, f"saliency_frame{i}.npy"), sal_np)
        
        avg_saliency_per_video = np.mean(saliency_accumulator, axis=0)
        np.save(os.path.join(out_dir, "avg_saliency_per_video.npy"), avg_saliency_per_video)
        # save_saliency_map(avg_saliency_per_video, f"Avg Spatial Saliency ({video_folder})",
        #                     os.path.join(out_dir, "avg_saliency_per_video.png"))
        save_saliency_map(avg_saliency_per_video, os.path.join(out_dir, "avg_saliency_per_video.png"))

        if global_saliency_storage is not None:
            global_saliency_storage.append(avg_saliency_per_video)

            # with torch.no_grad():
            #     unnorm = unnormalize_spatial(t_img[0]).cpu().numpy()
            #     unnorm = np.clip(unnorm, 0, 1).transpose(1, 2, 0)
            # plt.imsave(os.path.join(out_dir, f"frame{i}.png"), unnorm)

    else:  # temporal
        if len(frame_files) != 20:
            print(f"Expected 20 flow frames, but got {len(frame_files)}. Skipping temporal saliency.")
            return

        frames = []
        for frame_name in frame_files:
            img = Image.open(os.path.join(video_path, frame_name)).convert("L")
            t = transform(img)  # shape: [1, 224, 224]
            frames.append(t)

        flow_tensor = torch.cat(frames, dim=0)  # shape [20, 224, 224]
        flow_tensor = flow_tensor.unsqueeze(0).to(device).to(torch.float32)  # shape [1, 20, 224, 224]
        flow_tensor.requires_grad_()

        sal_flow = vanilla_saliency(model, flow_tensor, target_class=torch.tensor([0], device=device))[0].detach().cpu().numpy()

        for i in range(0, sal_flow.shape[0], 2):
            sal_pair = np.mean(sal_flow[i:i+2], axis=0)
            save_saliency_map(sal_pair, os.path.join(out_dir, f"saliency_{i}+{i+1}.png"))

            # for c in [0, 1]:
            #     flow_img = frames[i + c].squeeze(0) * 0.5 + 0.5  # unnormalize
            #     flow_img_np = flow_img.clamp(0, 1).cpu().numpy()
            #     save_image(flow_img_np, os.path.join(out_dir, f"flow_{i+c}.png"), cmap='gray')

        avg_saliency_per_video = np.mean(sal_flow, axis=0)
        np.save(os.path.join(out_dir, "avg_saliency_per_video.npy"), avg_saliency_per_video)
        save_saliency_map(avg_saliency_per_video, os.path.join(out_dir, "avg_saliency_per_video.png"))

        if global_saliency_storage is not None:
            global_saliency_storage.append(avg_saliency_per_video)

def save_average_across_videos(global_saliency_storage, save_dir):
    avg_across_videos = np.mean(global_saliency_storage, axis=0)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "avg_saliency_all_videos.npy"), avg_across_videos)
    # save_saliency_map(avg_across_videos, "Average Saliency Across All Videos",
    #                   os.path.join(save_dir, "avg_saliency_all_videos.png"))
    save_saliency_map(avg_across_videos, os.path.join(save_dir, "avg_saliency_all_videos.png"))    

def load_test_video_list(test_list_path):
    with open(test_list_path, 'r') as f:
        video_names = [line.strip().split()[0] for line in f if line.strip()]
    return video_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stream Explanations")
    parser.add_argument('--use_pruned', action='store_true', help='Use pruned spatial and temporal models')
    parser.add_argument('--run_id', type=str, required=True, help='Model run id (e.g., run1)')
    parser.add_argument('--prune_amount', type=str, default=None,
                        help='Amount of pruning (e.g., 4_00) — required if using pruned models')
    args = parser.parse_args()
    if args.use_pruned and not args.prune_amount:
        parser.error("--prune_amount is required when using pruned models")
    model_variant = "pruned" if args.use_pruned else "unpruned"
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spatial_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    temporal_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
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

    spatial_model = load_spatial_model_dropout(101, SPATIAL_MODEL_PATH, device)
    temporal_model = load_temporal_model(101, TEMPORAL_MODEL_PATH, device)
    spatial_data_dir = os.path.join(BASE_DIR, "data", "extracted_rgb_frames")
    test_list_path = os.path.join(BASE_DIR, "data", "splits", "testlist03_processed.txt")
    temporal_data_dir = os.path.join(BASE_DIR, "data", "extracted_optical_flow_frames")

    test_videos = load_test_video_list(test_list_path)
    global_saliency = []
    global_temporal_saliency = []

    for video_folder in test_videos:
        full_video_path = os.path.join(spatial_data_dir, video_folder)
        if not os.path.isdir(full_video_path):
            print(f"Warning: Folder '{full_video_path}' not found. Skipping.")
            continue

        process_video(video_folder, 'spatial', spatial_model, spatial_transform, spatial_data_dir, device, global_saliency)
        process_video(video_folder, 'temporal', temporal_model, temporal_transform, temporal_data_dir, device, global_temporal_saliency)

    save_average_across_videos(global_saliency, os.path.join(BASE_DIR, "plots", "saliency_maps", f"{args.run_id}", f"{args.prune_amount}", "spatial" "spatial_summary"))
    save_average_across_videos(global_temporal_saliency, os.path.join(BASE_DIR, "plots", "saliency_maps", f"{args.run_id}", f"{args.prune_amount}", "temporal", "temporal_summary"))
    
    # Test only one specific video
    # test_video = "v_ApplyEyeMakeup_g15_c01"  # Replace with an actual folder name

    # global_saliency = []

    # full_video_path = os.path.join(spatial_data_dir, test_video)
    # if not os.path.isdir(full_video_path):
    #     print(f"Folder '{full_video_path}' not found. Please check the name.")
    # else:
    #     process_video(test_video, 'spatial', spatial_model, spatial_transform, spatial_data_dir, device, global_saliency)
