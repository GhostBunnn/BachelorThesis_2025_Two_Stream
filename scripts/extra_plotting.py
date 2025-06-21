import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

video_folder = "spatial_v_CliffDiving_g18_c04"
video_rgb = "v_CliffDiving_g18_c04"
frames_dir = os.path.join(BASE_DIR, "data", "extracted_rgb_frames", video_rgb)
print(frames_dir)
overlay_dir = os.path.join(BASE_DIR, "plots", "overlay_outputs", video_folder)
irof_path = os.path.join(BASE_DIR, "plots", "irof_results", video_folder, 'irof_scores.csv')
savefig_path = os.path.join(BASE_DIR, "plots", "irof_plots")

irof_data = pd.read_csv(irof_path, header=None, names=["frame_file", "irof_auc"])
irof_data["irof_auc"] = irof_data["irof_auc"].astype(str).str.replace(",", " ").astype(float)

irof_array = pd.to_numeric(np.asarray(irof_data["irof_auc"]), errors='coerce')
nr_rows = 2

frame_files = sorted([
    f for f in os.listdir(frames_dir)
    if f.endswith(".jpg") or f.endswith(".png")
])
nr_frames = sum(1 for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png"))

fig, ax = plt.subplots(nr_rows, nr_frames, figsize=(4*nr_frames, 4*nr_rows), constrained_layout=True)
# fig, ax = plt.subplots(nr_rows, nr_frames, figsize=(4*nr_frames, 4*nr_rows))

for i, frame_file in enumerate(frame_files):
    # Load images
    frame_img = Image.open(os.path.join(frames_dir, frame_file))
    overlay_img = Image.open(os.path.join(overlay_dir, f"overlay_frame{i}.png"))

    # Show original frame
    im = ax[0, i].imshow(frame_img)
    ax[0, i].set_title(f"IROF = {irof_array[i]}")
    ax[0, i].xaxis.set_visible(False)

    # Show overlay
    ax[1, i].imshow(overlay_img)
    if i!= 0:
        ax[0, i].yaxis.set_visible(False)
        ax[1, i].yaxis.set_visible(False)
        
norm = plt.Normalize(vmin=0, vmax = 1)
sm = cm.ScalarMappable(cmap='jet', norm=norm)
sm.set_array([])

char = fig.colorbar(sm, ax=ax.ravel().tolist(), shrink=0.6, aspect=50, pad=0.02)
char.set_label('Saliency Intensity')

# plt.tight_layout()
plt.suptitle(f"Average IROF = {np.mean(irof_array):.6f}")
plt.savefig(savefig_path + '/' + video_rgb + "_irof_combined.png")
