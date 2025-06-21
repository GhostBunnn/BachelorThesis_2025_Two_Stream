import sys
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pandas as pd
from scipy.stats import wilcoxon
import pingouin as pg

# Path to your CSV file
csv_path = os.path.join(BASE_DIR, "results", "pruning_accuracies.csv")

# Load and clean data
df = pd.read_csv(csv_path, names=["run_id", "prune_percent", "accuracy", "label"])
df = df[df["label"].isin(["baseline unpruned", "pruned"])]  # Filter if needed
df["prune_percent"] = df["prune_percent"].astype(float)

# Pivot: one row per run, one column per pruning level
pivot_df = df.pivot(index="run_id", columns="prune_percent", values="accuracy").sort_index(axis=1)
print("\n Pivoted Accuracy Table:")
print(pivot_df)

# -----------------------
# Repeated-Measures ANOVA
# -----------------------
long_df = pivot_df.reset_index().melt(id_vars=["run_id"], var_name="prune_percent", value_name="accuracy")
long_df["prune_percent"] = long_df["prune_percent"].astype(str)

print("\n Running Repeated-Measures ANOVA...")
anova = pg.rm_anova(dv="accuracy", within="prune_percent", subject="run_id", data=long_df, detailed=True)
print(anova)
anova_out_path = os.path.join(BASE_DIR, "results", "anova_results.csv")
anova.to_csv(anova_out_path, index=False)
print(f"\nANOVA table saved to: {anova_out_path}")


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
        wilcoxon_results.append([col, None, f"Test failed: {e}"])

# Save to CSV
wilcoxon_df = pd.DataFrame(wilcoxon_results, columns=["prune_percent", "W_statistic", "p_value"])
wilcoxon_out_path = os.path.join(BASE_DIR, "results", "wilcoxon_results.csv")
wilcoxon_df.to_csv(wilcoxon_out_path, index=False)
print(f"\nWilcoxon results saved to: {wilcoxon_out_path}")


plt.figure(figsize=(8, 5))
mean_accuracies = pivot_df.mean()
baseline = mean_accuracies[0.0]
drops = baseline - mean_accuracies

plt.plot(drops.index, drops.values, marker='o')
plt.title("Accuracy Drop vs. Pruning Percentage")
plt.xlabel("Pruning Percentage (%)")
plt.ylabel("Drop from Baseline Accuracy (%)")
plt.grid(True)

plot_path = os.path.join(BASE_DIR, "results", "accuracy_drop_plot.png")
plt.savefig(plot_path)
print(f"\nAccuracy drop plot saved to: {plot_path}")

