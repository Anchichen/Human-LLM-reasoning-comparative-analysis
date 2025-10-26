# analyse_transition_patterns.py

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# === Config Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "analysis_data" / "full_human_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis_results" / "human" / "transition_patterns"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Load and Preprocess ===
df = pd.read_csv(DATA_PATH)
df = df.sort_values(by=["ppt_id", "task_id", "thought_id"])

# Generate transitions
df["next_state"] = df.groupby(["ppt_id", "task_id"])["high_level_code"].shift(-1)
df["valid_transition"] = df["high_level_code"].notna() & df["next_state"].notna()
transitions = df[df["valid_transition"]].copy()
transitions["transition"] = transitions["high_level_code"] + " → " + transitions["next_state"]

# === 1. Overall Transition Frequencies ===
overall_freq = transitions["transition"].value_counts().reset_index()
overall_freq.columns = ["transition", "count"]
overall_freq.to_csv(OUTPUT_DIR / "overall_transition_frequencies.csv", index=False)

# === 2. Frequency Comparison by Accuracy Group ===
grouped = transitions.groupby(["transition", "accuracy"]).size().unstack(fill_value=0)
grouped.columns = ["Incorrect", "Correct"]
grouped["Total"] = grouped["Correct"] + grouped["Incorrect"]
grouped["Correct_prob"] = grouped["Correct"] / grouped["Total"]
grouped["Incorrect_prob"] = grouped["Incorrect"] / grouped["Total"]

# Z-score and p-value
p1 = grouped["Correct"] / grouped["Total"]
p2 = grouped["Incorrect"] / grouped["Total"]
p = (grouped["Correct"] + grouped["Incorrect"]) / (2 * grouped["Total"])
se = np.sqrt(p * (1 - p) * 2 / grouped["Total"])
grouped["z_score"] = (p1 - p2) / se
grouped["p_value"] = 2 * (1 - norm.cdf(np.abs(grouped["z_score"])))
grouped["significant"] = grouped["p_value"] < 0.05
grouped = grouped.reset_index()
grouped.to_csv(OUTPUT_DIR / "transition_comparison_correct_vs_incorrect.csv", index=False)

# === 3. Gini Coefficient per Participant ===
ppt_transitions = transitions.groupby(["ppt_id", "transition"]).size().unstack(fill_value=0)

def gini(array):
    sorted_arr = np.sort(array)
    n = len(array)
    if np.sum(array) == 0:
        return 0
    return (2 * np.sum((np.arange(1, n + 1) * sorted_arr)) / (n * np.sum(sorted_arr))) - (n + 1) / n

ppt_gini = ppt_transitions.apply(gini, axis=1).reset_index()
ppt_gini.columns = ["ppt_id", "gini_coefficient"]
ppt_gini["gini_bin"] = pd.qcut(ppt_gini["gini_coefficient"], q=3, labels=["Balanced", "Moderate", "Unbalanced"])
ppt_gini.to_csv(OUTPUT_DIR / "transition_gini_scores.csv", index=False)

# Merge for transition binning
transitions = transitions.merge(ppt_gini[["ppt_id", "gini_bin"]], on="ppt_id", how="left")
gini_trans = transitions.groupby(["gini_bin", "transition"]).size().reset_index(name="count")
gini_trans = gini_trans.sort_values(["gini_bin", "count"], ascending=[True, False])
gini_trans.to_csv(OUTPUT_DIR / "transition_by_gini_bin.csv", index=False)

# === Plots ===
sns.set(style="whitegrid")

# 1. Top overall transitions
plt.figure(figsize=(10, 6))
top_trans = overall_freq.head(15)
sns.barplot(data=top_trans, y="transition", x="count", palette="viridis")
plt.title("Top 15 Most Frequent Transition Pairs")
plt.xlabel("Count")
plt.ylabel("Transition")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "barplot_top_transitions.png")
plt.close()

# 2. Top transitions: Correct vs Incorrect
top_comp = grouped.sort_values("Total", ascending=False).head(15)
top_comp_melted = top_comp.melt(id_vars=["transition"], value_vars=["Correct", "Incorrect"], 
                                var_name="Group", value_name="Count")
plt.figure(figsize=(12, 6))
sns.barplot(data=top_comp_melted, x="Count", y="transition", hue="Group", palette="Set2")
plt.title("Top 15 Transitions: Correct vs Incorrect Group")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "barplot_correct_vs_incorrect_transitions.png")
plt.close()

# 3. Transitions per Gini bin
plt.figure(figsize=(12, 6))
top_gini_trans = gini_trans.groupby("gini_bin").head(10)
sns.barplot(data=top_gini_trans, x="count", y="transition", hue="gini_bin", dodge=False)
plt.title("Top Transitions Within Each Gini Bin (Balanced → Unbalanced)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "barplot_transitions_by_gini_bin.png")
plt.close()

print("✅ Transition pattern analyses complete. Outputs saved in:", OUTPUT_DIR)
