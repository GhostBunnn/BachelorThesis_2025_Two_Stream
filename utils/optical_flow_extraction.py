import os
import cv2
import torch
import torchvision
import numpy as np
from pathlib import Path
from torchvision.transforms.functional import to_tensor, pad
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
from tqdm import tqdm  # For progress tracking

# Load RAFT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()


def pad_to_divisible_by_8(tensor):
    """
    Pads a tensor to make its height and width divisible by 8.

    Args:
        tensor (torch.Tensor): Input tensor of shape (1, C, H, W).
    
    Returns:
        padded_tensor (torch.Tensor): Tensor padded to dimensions divisible by 8.
        padding (tuple): Padding applied (left, right, top, bottom).
    """
    _, _, h, w = tensor.shape
    pad_h = (8 - h % 8) if h % 8 != 0 else 0
    pad_w = (8 - w % 8) if w % 8 != 0 else 0
    padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    padded_tensor = pad(tensor, padding)
    return padded_tensor, padding


def unpad(tensor, padding):
    """
    Removes the padding from a tensor.

    Args:
        tensor (torch.Tensor): Padded tensor of shape (1, C, H, W).
        padding (tuple): Padding applied (left, right, top, bottom).

    Returns:
        unpadded_tensor (torch.Tensor): Tensor with padding removed.
    """
    _, _, h, w = tensor.shape
    left, right, top, bottom = padding
    return tensor[..., top:h-bottom, left:w-right] if (top + bottom + left + right) > 0 else tensor


def compute_optical_flow(model, frame1_path, frame2_path):
    # Read and preprocess RGB frames to match pytorch model requirements
    frame1 = to_tensor(cv2.imread(frame1_path)).unsqueeze(0).to(device)
    frame2 = to_tensor(cv2.imread(frame2_path)).unsqueeze(0).to(device)

    # Pad frames to match the size required by RAFT
    frame1_padded, pad1 = pad_to_divisible_by_8(frame1)
    frame2_padded, pad2 = pad_to_divisible_by_8(frame2)

    # Compute optical flow
    with torch.no_grad():
        flow_list = model(frame1_padded, frame2_padded)  # RAFT returns a list of flow tensors

    # Get the highest-resolution flow (first element in the list)
    flow = flow_list[0]

    # Unpad the flow to match the original frame size
    flow = unpad(flow, pad1)

    # Separate the horizontal (x) and vertical (y) components of the flow
    flow_x, flow_y = flow[0, 0].cpu().numpy(), flow[0, 1].cpu().numpy()
    return flow_x, flow_y


def process_video_frames(input_dir, output_dir):
    """
    Compute and save optical flow for consecutive frames in a video.

    Args:
        input_dir (str): Directory containing extracted RGB frames.
        output_dir (str): Directory to save computed optical flow images.
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_files = sorted(os.listdir(input_dir))  # Ensure frames are processed in order

    total_frames = len(frame_files) - 1  # Total number of frame pairs
    num_samples = 10

    # Uniformly sample 10 frame pairs
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    for i, idx in enumerate(indices):
        frame1_path = os.path.join(input_dir, frame_files[idx])
        frame2_path = os.path.join(input_dir, frame_files[idx + 1])

        flow_x, flow_y = compute_optical_flow(model, frame1_path, frame2_path)

        flow_x_path = os.path.join(output_dir, f"flow_x_{i:04d}.jpg")
        flow_y_path = os.path.join(output_dir, f"flow_y_{i:04d}.jpg")
        cv2.imwrite(flow_x_path, cv2.normalize(flow_x, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))
        cv2.imwrite(flow_y_path, cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))


def process_all_videos(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)

    video_dirs = list(input_root.iterdir())
    total_videos = len(video_dirs)

    with tqdm(total=total_videos, desc="Processing all videos", unit="video") as pbar:
        for video_dir in video_dirs:
            if video_dir.is_dir():
                process_video_frames(video_dir, output_root / video_dir.name)
                pbar.update(1)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    EXTRACTED_FRAMES_PATH = BASE_DIR / "data" / "extracted_rgb_frames"
    OPTICAL_FLOW_PATH = BASE_DIR / "data" / "extracted_optical_flow_frames"

    process_all_videos(EXTRACTED_FRAMES_PATH, OPTICAL_FLOW_PATH)
