import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

base_font_size = 24
plt.rc('font', size = base_font_size)
plt.rc('axes', linewidth=5, titlesize = base_font_size+2, labelsize = base_font_size+2)
plt.rc('xtick', top=True, bottom=True, direction='in')
plt.rc('ytick', left=True, right=True, direction='in') 
plt.rc('figure', titlesize=base_font_size+4, dpi=400)
plt.rc('legend', fontsize=base_font_size-1, title_fontsize=base_font_size-1, frameon=False)
plt.rc('lines', linewidth=5, markersize = 12)

# Folder where all CSVs are stored
data_dir = os.path.join(BASE_DIR, "results")
stream_type = ['spatial', 'temporal']

for stream in stream_type:
    data = pd.read_csv(os.path.join(data_dir, f"unpruned_data_{stream}.csv"))
    epoch = pd.to_numeric(np.asarray(data['epoch']), errors='coerce')
    trainacc = pd.to_numeric(np.asarray(data['avg_trainacc']), errors='coerce')
    trainloss = pd.to_numeric(np.asarray(data['avg_trainloss']), errors='coerce')
    valacc = pd.to_numeric(np.asarray(data['avg_valacc']), errors='coerce')
    valloss = pd.to_numeric(np.asarray(data['avg_valloss']), errors='coerce')
    
    
    output_path=os.path.join(BASE_DIR, "results", f"unpruned_{stream}_results.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10,12), sharex=True)
    ax1.plot(epoch, trainacc, label = f'Training ({stream})', color = 'blue', marker='o')
    ax1.plot(epoch, valacc, label = f'Validation ({stream})', color = 'red', marker='o')
    ax2.plot(epoch, trainloss, label = f'Training ({stream})', color = 'blue', marker='o')
    ax2.plot(epoch, valloss, label = f'Validation ({stream})', color = 'red', marker='o')
    ax2.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy [%]")
    ax2.set_ylabel("Loss")
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()