import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.patches as mpatches

base_font_size = 20
plt.rc('font', size = base_font_size)
plt.rc('axes', linewidth=3, titlesize = base_font_size+2, labelsize = base_font_size+2)
plt.rc('xtick', top=True, bottom=True, direction='in')
plt.rc('ytick', left=True, right=True, direction='in') 
plt.rc('figure', titlesize=base_font_size+4, dpi=300)
plt.rc('legend', fontsize=base_font_size-1, title_fontsize=base_font_size-1, frameon=False)
plt.rc('lines', linewidth=3)

# plotting function
def plot_function(df, output_path, x_axis, y_axis, figsize=(12, 18), hspace=0.1):
    spatial_df = df[df["stream"] == "spatial"]
    temporal_df = df[df["stream"] == "temporal"]
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1]}, figsize=figsize)
    fig.subplots_adjust(hspace=hspace)

    if y_axis.startswith('Wilcoxon') or y_axis.startswith('Cohen'):
        ax1.bar(spatial_df["prune_percent"], spatial_df["value"], color='orange', label='Spatial')
        ax2.bar(temporal_df["prune_percent"], temporal_df["value"], color='blue', label='Temporal')
        if y_axis.startswith('Wilcox'):
            ax1.axhline(0.05, color='red', linestyle='--', label='Significance Threshold (p = 0.05)')
            ax2.axhline(0.05, color='red', linestyle='--', label='Significance Threshold (p = 0.05)')
        else:
            ax1.axhline(0.2, color='gray', linestyle='--', label='Small Effect (0.2)')
            ax1.axhline(0.5, color='orange', linestyle='--', label='Medium Effect (0.5)')
            ax1.axhline(0.8, color='green', linestyle='--', label='Large Effect (0.8)')
            ax2.axhline(0.2, color='gray', linestyle='--', label='Small Effect (0.2)')
            ax2.axhline(0.5, color='orange', linestyle='--', label='Medium Effect (0.5)')
            ax2.axhline(0.8, color='green', linestyle='--', label='Large Effect (0.8)')
    elif y_axis.startswith('Mean'):      
        ax1.errorbar(spatial_df["prune_percent"], spatial_df["mean_accuracy"], yerr=spatial_df["std_dev"], fmt='-o', markersize=12, capsize=10, color='orange', label = 'Spatial')
        ax2.errorbar(temporal_df["prune_percent"], temporal_df["mean_accuracy"], yerr=temporal_df["std_dev"], fmt='-o', markersize=12, capsize=10, color='blue', label = 'Temporal')

    ax2.set_xticks(temporal_df["prune_percent"])
    ax2.set_xticklabels([f"{x:.2f}" for x in temporal_df["prune_percent"]], rotation=90)

    ax2.set_xlabel(x_axis, labelpad=20)
    ax1.set_ylabel(y_axis)
    ax2.set_ylabel(y_axis)

    ax1.legend(alignment='left')
    ax2.legend(alignment='left')
    ax1.grid(True)
    ax2.grid(True)

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()
    
# combine csvs
def load_and_prepare_data(spatial_path, temporal_path):
    spatial_df = pd.read_csv(spatial_path, skiprows=1, names=["run_id", "prune_percent", "accuracy", "label"])
    temporal_df = pd.read_csv(temporal_path, skiprows=1, names=["run_id", "prune_percent", "accuracy", "label"])

    spatial_df["stream"] = "spatial"
    temporal_df["stream"] = "temporal"
    combined_df = pd.concat([spatial_df, temporal_df], ignore_index=True)

    combined_df = combined_df[combined_df["label"].isin(["baseline unpruned", "pruned", "unpruned"])]
    
    # Ensure prune_percent is a valid float
    combined_df["prune_percent"] = pd.to_numeric(combined_df["prune_percent"], errors='coerce')
    combined_df = combined_df.dropna(subset=["prune_percent"])

    pivot_dfs = {
        stream: df[df["stream"] == stream].pivot_table(
            index="run_id", columns="prune_percent", values="accuracy", aggfunc="mean"
        ).sort_index(axis=1)
        for stream, df in combined_df.groupby("stream")
    }

    return combined_df, pivot_dfs


# compute wilcoxon
def run_wilcoxon_by_stream(df, pivot_dfs, baseline_col=0.0):
    results = []

    for stream in df["stream"].unique():
        pivot_df = pivot_dfs[stream]
        for col in pivot_df.columns:
            if col == baseline_col:
                continue
            try:
                stat, p = wilcoxon(pivot_df[baseline_col], pivot_df[col])
                results.append({
                    "stream": stream,
                    "prune_percent": col,
                    "W_statistic": stat,
                    "value": p
                })
            except ValueError as e:
                print(f"Wilcoxon test failed for stream={stream}, prune={col}: {e}")
    
    return pd.DataFrame(results).dropna()

def plot_boxplots_irof(df, output_dir="results", prefix=""):
    os.makedirs(output_dir, exist_ok=True)
    for stream in df["stream"].unique():
        
        stream_df = df[df["stream"] == stream]
        pivot_df = stream_df.pivot_table(index="run_id", columns="prune_percent", values="accuracy", aggfunc="mean")

        melt_df = pivot_df.reset_index().melt(id_vars="run_id", var_name="prune_percent", value_name="accuracy")
        melt_df["prune_percent"] = melt_df["prune_percent"].apply(lambda x: f"{float(x):.2f}")

        plt.figure(figsize=(16, 8))
        ax = sns.boxplot(
            data=melt_df,
            x="prune_percent",
            y="accuracy",
            color='blue',
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 8},
            medianprops={"color": "orange", "linewidth": 3}
        )
        plt.xticks(rotation=90)

        mean_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='Mean (white dot)')
        median_patch = mpatches.Patch(color='orange', label='Median (orange line)')
        legend_loc = 'upper left' if stream == 'spatial' else 'lower left'
        plt.legend(handles=[mean_patch, median_patch], loc=legend_loc)

        ax.set_xlabel("Pruning Percentage")
        ax.set_ylabel("IROF AUC")

        plt.grid(True, linestyle='--', alpha=0.5)
        out_path = os.path.join(output_dir, f"{prefix}{stream}_boxplot_wilcoxon.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"IROF boxplot saved to: {out_path}")


def plot_boxplots_with_wilcoxon(df, baseline_col=0.0, output_dir="results", prefix=""):
    os.makedirs(output_dir, exist_ok=True)
    for stream in df["stream"].unique():
        stream_df = df[df["stream"] == stream]
        pivot_df = stream_df.pivot_table(index="run_id", columns="prune_percent", values="accuracy", aggfunc="mean")
        baseline_vals = pivot_df[baseline_col]

        melt_df = pivot_df.reset_index().melt(id_vars="run_id", var_name="prune_percent", value_name="accuracy")
        melt_df["prune_percent"] = melt_df["prune_percent"].apply(lambda x: f"{float(x):.2f}")

        plt.figure(figsize=(16, 8))
        ax = sns.boxplot(
            data=melt_df,
            x="prune_percent",
            y="accuracy",
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 8},
            medianprops={"color": "orange", "linewidth": 3}
        )
        plt.xticks(rotation=90)
        mean_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='Mean (white dot)')
        median_patch = mpatches.Patch(color='orange', label='Median (orange line)')
        if stream == 'spatial':
            legend_loc = 'upper left'
        else:
            legend_loc = 'lower left'
        plt.legend(handles=[mean_patch, median_patch], loc = legend_loc)
        ax.set_xlabel("Pruning Percentage")
        ax.set_ylabel("Accuracy")

        # Annotate Wilcoxon p-values
        for tick, prune_str in enumerate(sorted(melt_df["prune_percent"].unique(), key=lambda x: float(x))):
            prune = float(prune_str)
            if prune == baseline_col or prune not in pivot_df.columns:
                continue
            try:
                stat, p = wilcoxon(baseline_vals, pivot_df[prune])
                p_label = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
                ax.text(tick, ax.get_ylim()[1]*0.98, p_label, ha='center', va='top', fontsize=12, color='red', weight='bold')
            except Exception as e:
                print(f"Could not compute Wilcoxon for {stream}, prune={prune}: {e}")

        plt.grid(True, linestyle='--', alpha=0.5)
        out_path = os.path.join(output_dir, f"{prefix}{stream}_boxplot_wilcoxon.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Boxplot saved to: {out_path}")


# compute cohen's d
def compute_cohens_d_by_stream(df, pivot_dfs, baseline_col=0.0):
    def cohens_d(x, y):
        diff = x - y
        return diff.mean() / diff.std(ddof=1)

    results = []

    for stream in df["stream"].unique():
        pivot_df = pivot_dfs[stream]
        baseline_scores = pivot_df[baseline_col]

        for col in pivot_df.columns:
            if col == baseline_col:
                continue
            try:
                d = cohens_d(pivot_df[col], baseline_scores)
                results.append({
                    "stream": stream,
                    "prune_percent": col,
                    "value": d
                })
            except Exception as e:
                print(f"Cohen's d failed for stream={stream}, prune={col}: {e}")
    
    return pd.DataFrame(results).dropna()


def compute_mean_accuracy_by_stream(pivot_dfs):
    summary_rows = []

    for stream, pivot_df in pivot_dfs.items():
        mean_accuracies = pivot_df.mean()
        std_accuracies = pivot_df.std()

        for prune_percent in mean_accuracies.index:
            summary_rows.append({
                "stream": stream,
                "prune_percent": prune_percent,
                "mean_accuracy": mean_accuracies[prune_percent],
                "std_dev": std_accuracies[prune_percent]
            })

    return pd.DataFrame(summary_rows).sort_values(by=["stream", "prune_percent"])


#####################################
# begin here
#####################################
spatial_csv_path = os.path.join(BASE_DIR, "results", "spatial_pruning_accuracies.csv")
temporal_csv_path = os.path.join(BASE_DIR, "results", "temporal_pruning_accuracies.csv")

combined_df, pivot_df = load_and_prepare_data(spatial_csv_path, temporal_csv_path)
wilcoxon_df = run_wilcoxon_by_stream(combined_df, pivot_df)
plot_boxplots_with_wilcoxon(combined_df, baseline_col=0.0, output_dir=os.path.join(BASE_DIR, "results"), prefix="pruning_")
cohens_d_df = compute_cohens_d_by_stream(combined_df, pivot_df)
mean_accuracy_df = compute_mean_accuracy_by_stream(pivot_df)


#####################################
# Wilcoxon 
#####################################

for stream in wilcoxon_df["stream"].unique():
    stream_df = wilcoxon_df[wilcoxon_df["stream"] == stream]
    out_path = os.path.join(BASE_DIR, "results", f"{stream}_wilcoxon_results.csv")
    stream_df.to_csv(out_path, index=False)
    print(f"Wilcoxon results saved to: {out_path}")

# Plot
wilcoxon_plot_path = os.path.join(BASE_DIR, "results", "wilcoxon_pruning_plot.png")
plot_function(wilcoxon_df, wilcoxon_plot_path, "Pruning Ratio [%]", "Wilcoxon p-values")

#####################################
# Cohen's d
#####################################

# Save per stream
for stream in cohens_d_df["stream"].unique():
    stream_df = cohens_d_df[cohens_d_df["stream"] == stream]
    out_path = os.path.join(BASE_DIR, "results", f"{stream}_cohens_d_results.csv")
    stream_df.to_csv(out_path, index=False)
    print(f"Cohen's d results saved to: {out_path}")

# Plot
cohen_plot_path = os.path.join(BASE_DIR, "results", "cohens_d_plot.png")
plot_function(cohens_d_df, cohen_plot_path, x_axis="Pruning Ratio [%]", y_axis="Cohen's d")

#####################################
# Mean Accuracy vs Pruning Percentage (with Std Dev)
#####################################

# Save per stream
for stream in mean_accuracy_df["stream"].unique():
    stream_df = mean_accuracy_df[mean_accuracy_df["stream"] == stream]
    out_path = os.path.join(BASE_DIR, "results", f"{stream}_mean_accuracy.csv")
    stream_df.to_csv(out_path, index=False)
    print(f"Mean accuracy summary saved to: {out_path}")

mean_accuracy_plot_path = os.path.join(BASE_DIR, "results", "mean_accuracy_plot.png")
plot_function(mean_accuracy_df, mean_accuracy_plot_path, x_axis="Pruning Ratio [%]", y_axis="Mean Accuracy [%]")

####################################
# Accuracy trend
####################################

run_plot_path = os.path.join(BASE_DIR, "results", "per_run_accuracy_plot.png")

spatial_df = combined_df[combined_df["stream"] == "spatial"]
temporal_df = combined_df[combined_df["stream"] == "temporal"]
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1]}, figsize=(12, 18))
fig.subplots_adjust(hspace=0.1)

for run_id, row in pivot_df["spatial"].iterrows():
    ax1.plot(row.index, row.values, marker='o', markersize=12, label=f"{run_id}")
for run_id, row in pivot_df["temporal"].iterrows():
    ax2.plot(row.index, row.values, marker='o', markersize=12, label=f"{run_id}")

ax2.set_xticks(temporal_df["prune_percent"])
ax2.set_xticklabels([f"{x:.2f}" for x in temporal_df["prune_percent"]], rotation=90)

ax2.set_xlabel("Pruning Ratio [%]", labelpad=20)
ax1.set_ylabel("Accuracy [%]")
ax2.set_ylabel("Accuracy [%]")

ax1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), alignment='center', frameon=True, title='Spatial')
ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), alignment='center', frameon=True, title='Temporal')
ax1.grid(True)
ax2.grid(True)

plt.savefig(run_plot_path, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()

####################################
# IROF boxplots
####################################
irof_dir = os.path.join(BASE_DIR, "plots", "irof_plots")

def parse_prune_ratio(prune_str):
    if prune_str == "unpruned":
        return 0.0
    return float(prune_str.replace("percent", "").replace("_", "."))

# Collect CSV paths
irof_csv_paths = []
for run_id in ["run1", "run2", "run3", "run4", "run5"]:
    run_dir = os.path.join(irof_dir, run_id)
    if not os.path.exists(run_dir):
        continue
    for prune_ratio in os.listdir(run_dir):
        for stream in ["spatial", "temporal"]:
            csv_path = os.path.join(run_dir, prune_ratio, stream, f"all_{stream}_irof_scores.csv")
            if os.path.exists(csv_path):
                irof_csv_paths.append((run_id, prune_ratio, stream, csv_path))

# Build a combined dataframe
irof_records = []
for run_id, prune_ratio, stream, csv_path in irof_csv_paths:
    prune_val = parse_prune_ratio(prune_ratio)
    df = pd.read_csv(csv_path)
    # CSV has: video, frame_file, irof_auc
    for _, row in df.iterrows():
        irof_records.append({
            "run_id": run_id,
            "prune_percent": prune_val,
            "accuracy": row["irof_auc"],
            "stream": stream
        })

irof_df = pd.DataFrame(irof_records)
plot_boxplots_irof(irof_df, output_dir=os.path.join(BASE_DIR, "results"), prefix="irof_")
