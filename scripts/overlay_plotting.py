import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

'''
This script only works for specific videos!!!
'''

def parse_prune_ratio(prune_str):
    if prune_str == "unpruned":
        return 0.0
    return float(prune_str.replace("percent", "").replace("_", "."))

# plotting settings
base_font_size = 20
plt.rc('font', size = base_font_size)
plt.rc('axes', linewidth=3, titlesize = base_font_size+2, labelsize = base_font_size+2)
plt.rc('xtick', top=True, bottom=True, direction='in')
plt.rc('ytick', left=True, right=True, direction='in') 
plt.rc('figure', titlesize=base_font_size+4, dpi=300)
plt.rc('legend', fontsize=base_font_size-1, title_fontsize=base_font_size-1, frameon=False)
plt.rc('lines', linewidth=3)

video_folder = "spatial_v_Skijet_g16_c04"
video_rgb = "v_Skijet_g16_c04"
savefig_path = os.path.join(BASE_DIR, "plots", "irof_plots")
frames_dir = os.path.join(BASE_DIR, "data", "extracted_rgb_frames", video_rgb)
run_path = os.path.join(BASE_DIR, "plots", "overlay_outputs", "run1")
pruning_dir_list = [x for x in  os.listdir(run_path) if os.path.isdir(os.path.join(run_path, x))]
nr_rows = 2 + len(pruning_dir_list)
pruning_dir_list.sort(key=lambda z: (z != "unpruned", z))
overlay_dir_list = []
for entry in pruning_dir_list:
    overlay_dir_list.append(os.path.join(run_path,entry))

frame_files = sorted([
    f for f in os.listdir(frames_dir)
    if f.endswith(".jpg") or f.endswith(".png")
])
nr_frames = sum(1 for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png"))

fig, ax = plt.subplots(nr_rows, nr_frames, figsize=(4*nr_frames, 4*nr_rows), constrained_layout=True)

for i, frame_file in enumerate(frame_files):
    
    # Load images
    frame_img = Image.open(os.path.join(frames_dir, frame_file))
    
    # Show original frame
    ax[0, i].imshow(frame_img)
    
    # Show resized frame
    resized_frame = frame_img.resize((224, 224), resample=Image.BILINEAR)
    ax[1, i].imshow(resized_frame)
    
    for j, overlay_dir in enumerate(overlay_dir_list):  
        overlay_img = Image.open(os.path.join(overlay_dir, "spatial", f"{video_folder}", f"overlay_frame_000{i}.jpg"))

        # Show overlay
        ax[j+2, i].imshow(overlay_img)

prune_ratios = []
for y in pruning_dir_list:        
    prune_ratios.append(parse_prune_ratio(y))
prune_ratios = sorted(prune_ratios)
for r in range(nr_rows):
    for c in range(nr_frames):
        if c != 0:
            ax[r, c].yaxis.set_visible(False)
        if r != nr_rows-1:
            ax[r, c].xaxis.set_visible(False)
        if r == 0:
            ax[r,0].set_ylabel("Original Frame")
        elif r == 1:
            ax[r,0].set_ylabel("Resized Frame")
        else:
            ax[r,0].set_ylabel(f"Overlay - {prune_ratios[r-2]}%")
        
norm = plt.Normalize(vmin=0, vmax = 1)
sm = cm.ScalarMappable(cmap='jet', norm=norm)
sm.set_array([])

char = fig.colorbar(sm, ax=ax.ravel().tolist(), shrink=0.6, aspect=50, pad=0.02)
char.set_label('Saliency Intensity')

# plt.tight_layout()
plt.savefig(savefig_path + '/overlay_saliency_combined_' + video_rgb + ".png")
