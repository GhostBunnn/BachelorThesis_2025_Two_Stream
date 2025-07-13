import sys
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pandas as pd
from scipy.stats import wilcoxon
# import pingouin as pg

# Path to your CSV file
csv_path = os.path.join(BASE_DIR, "results", "spatial_pruning_accuracies.csv")
df = pd.read_csv(csv_path, skiprows=1, names=["run_id", "prune_percent", "accuracy", "label"])

# Keep only relevant labels
df = df[df["label"].isin(["baseline unpruned", "pruned", "unpruned"])]
df["prune_percent"] = df["prune_percent"].astype(float)

# Pivot data to wide format
pivot_df = df.pivot_table(index="run_id", columns="prune_percent", values="accuracy", aggfunc="mean").sort_index(axis=1)
print("\nPivoted Accuracy Table:")
print(pivot_df)


# -----------------------
# Wilcoxon Signed-Rank Tests (baseline vs each pruning level)
# -----------------------
baseline_col = 0.0
wilcoxon_results = []

for col in pivot_df.columns:
    if col == baseline_col:
        continue
    try:
        stat, p = wilcoxon(pivot_df[baseline_col], pivot_df[col])
        wilcoxon_results.append([col, stat, p])
    except ValueError as e:
        wilcoxon_results.append([col, None, None])
        print(f"Wilcoxon test failed for {col}: {e}")

# Convert to DataFrame
wilcoxon_df = pd.DataFrame(wilcoxon_results, columns=["prune_percent", "W_statistic", "p_value"])
wilcoxon_df = wilcoxon_df.dropna()  # Drop failed tests if any

# Save results
wilcoxon_out_path = os.path.join(BASE_DIR, "results", "wilcoxon_results.csv")
wilcoxon_df.to_csv(wilcoxon_out_path, index=False)
print(f"\nWilcoxon results saved to: {wilcoxon_out_path}")

# Plot p-values
plt.figure(figsize=(11, 7))
plt.bar(wilcoxon_df["prune_percent"], wilcoxon_df["p_value"], width=1.5)
plt.axhline(0.05, color='red', linestyle='--', label='Significance Threshold (p = 0.05)')
plt.xticks(wilcoxon_df["prune_percent"], labels=[f"{x:.2f}" for x in wilcoxon_df["prune_percent"]], rotation=45)
plt.xlabel("Pruning Percentage")
plt.ylabel("Wilcoxon Test p-value")
plt.title("Wilcoxon Test p-values vs. Pruning Percentage")
plt.legend()
plt.grid(True)

# Save plot
plot_path = os.path.join(BASE_DIR, "results", "wilcoxon_pvalues_plot.png")
plt.savefig(plot_path)
print(f"\nWilcoxon p-value plot saved to: {plot_path}")






# -----------------------
# Mean Accuracy vs Pruning Percentage (with Std Dev)
# -----------------------

# Calculate mean and std across runs for each pruning level
mean_accuracies = pivot_df.mean()
std_accuracies = pivot_df.std()

# Create a DataFrame for reporting
summary_df = pd.DataFrame({
    "prune_percent": mean_accuracies.index,
    "mean_accuracy": mean_accuracies.values,
    "std_dev": std_accuracies.values
}).sort_values("prune_percent")

# Save summary to CSV
summary_out_path = os.path.join(BASE_DIR, "results", "accuracy_summary.csv")
summary_df.to_csv(summary_out_path, index=False)
print(f"\nAccuracy summary saved to: {summary_out_path}")

# Plot mean accuracy with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(summary_df["prune_percent"], summary_df["mean_accuracy"],
             yerr=summary_df["std_dev"], fmt='-o', capsize=5)

# Set custom ticks to show exact pruning percentages
plt.xticks(summary_df["prune_percent"], labels=[f"{x:.2f}" for x in summary_df["prune_percent"]], rotation=45)
plt.title("Mean Accuracy ± Std Dev vs. Pruning Percentage")
plt.xlabel("Pruning Percentage (%)")
plt.ylabel("Mean Accuracy (%)")
plt.grid(True)

accuracy_plot_path = os.path.join(BASE_DIR, "results", "mean_accuracy_plot.png")
plt.savefig(accuracy_plot_path)
print(f"\nMean accuracy plot saved to: {accuracy_plot_path}")


# -----------------------
# Per-Run Accuracy Trends (Line Plot)
# -----------------------
plt.figure(figsize=(10, 6))
for run_id, row in pivot_df.iterrows():
    plt.plot(row.index, row.values, marker='o', label=f"{run_id}")

plt.xticks(summary_df["prune_percent"], labels=[f"{x:.2f}" for x in summary_df["prune_percent"]], rotation=45)
plt.title("Per-Run Accuracy vs. Pruning Percentage")
plt.xlabel("Pruning Percentage (%)")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title="Run ID")

run_plot_path = os.path.join(BASE_DIR, "results", "per_run_accuracy_plot.png")
plt.tight_layout()
plt.savefig(run_plot_path)
print(f"\nPer-run accuracy plot saved to: {run_plot_path}")


# -----------------------
# Cohen's d vs. Baseline
# -----------------------
def cohens_d(x, y):
    diff = x - y
    return diff.mean() / diff.std(ddof=1)

cohens_d_results = []
baseline_scores = pivot_df[baseline_col]

for col in pivot_df.columns:
    if col == baseline_col:
        continue
    try:
        d = cohens_d(pivot_df[col], baseline_scores)
        cohens_d_results.append([col, d])
    except Exception as e:
        cohens_d_results.append([col, None])
        print(f"Cohen's d failed for {col}: {e}")

cohens_d_df = pd.DataFrame(cohens_d_results, columns=["prune_percent", "cohens_d"])
cohens_d_df = cohens_d_df.dropna().sort_values("prune_percent")

# Save results
cohen_out_path = os.path.join(BASE_DIR, "results", "cohens_d_results.csv")
cohens_d_df.to_csv(cohen_out_path, index=False)
print(f"\nCohen's d results saved to: {cohen_out_path}")

# Plot Cohen's d
plt.figure(figsize=(10, 5))
plt.bar(cohens_d_df["prune_percent"], cohens_d_df["cohens_d"], width=1.5)
plt.axhline(0.2, color='gray', linestyle='--', label='Small Effect (0.2)')
plt.axhline(0.5, color='orange', linestyle='--', label='Medium Effect (0.5)')
plt.axhline(0.8, color='green', linestyle='--', label='Large Effect (0.8)')

plt.xticks(cohens_d_df["prune_percent"], labels=[f"{x:.2f}" for x in cohens_d_df["prune_percent"]], rotation=45)
plt.xlabel("Pruning Percentage")
plt.ylabel("Cohen's d (Effect Size)")
plt.title("Effect Size (Cohen's d) vs. Pruning Percentage")
plt.legend()
plt.grid(True)

cohen_plot_path = os.path.join(BASE_DIR, "results", "cohens_d_plot.png")
plt.tight_layout()
plt.savefig(cohen_plot_path)
print(f"\nCohen's d plot saved to: {cohen_plot_path}")






# -----------------------
# Repeated-Measures ANOVA
# -----------------------
# long_df = pivot_df.reset_index().melt(id_vars=["run_id"], var_name="prune_percent", value_name="accuracy")
# long_df["prune_percent"] = long_df["prune_percent"].astype(str)

# print("\n Running Repeated-Measures ANOVA...")
# anova = pg.rm_anova(dv="accuracy", within="prune_percent", subject="run_id", data=long_df, detailed=True)
# print(anova)
# anova_out_path = os.path.join(BASE_DIR, "results", "anova_results.csv")
# anova.to_csv(anova_out_path, index=False)
# print(f"\nANOVA table saved to: {anova_out_path}")




# # Save plot
# plot_path = os.path.join(BASE_DIR, "results", "wilcoxon_pvalues_plot.png")
# plt.savefig(plot_path)
# print(f"\nWilcoxon p-value plot saved to: {plot_path}")


# plt.figure(figsize=(8, 5))
# mean_accuracies = pivot_df.mean()
# baseline = mean_accuracies[0.0]
# drops = baseline - mean_accuracies

# plt.plot(drops.index, drops.values, marker='o')
# plt.title("Accuracy Drop vs. Pruning Percentage")
# plt.xlabel("Pruning Percentage (%)")
# plt.ylabel("Drop from Baseline Accuracy (%)")
# plt.grid(True)

# plot_path = os.path.join(BASE_DIR, "results", "accuracy_drop_plot.png")
# plt.savefig(plot_path)
# print(f"\nAccuracy drop plot saved to: {plot_path}")

