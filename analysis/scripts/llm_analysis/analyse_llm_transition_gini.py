
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import pairwise
from typing import List
from scipy.stats import skew

# === Config Paths ===
DATA_PATH = Path("../analysis_data/full_llm_data.csv")
OUTPUT_DIR = Path("../analysis_results/llm/transition_gini_combined")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Gini calculation function ===
def gini_coefficient(x: List[int]) -> float:
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0
    sorted_x = np.sort(np.array(x))
    n = len(x)
    cum_x = np.cumsum(sorted_x)
    return (n + 1 - 2 * np.sum(cum_x) / cum_x[-1]) / n

# === Load and clean data ===
df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"high-level code": "high_level_code"})  # in case the column name uses hyphen
df = df.dropna(subset=["ppt_id", "task_id", "high_level_code"])
df = df.sort_values(by=["ppt_id", "task_id", "thought_id"])

# --- Generate transitions within each ppt_id and task_id ---
df["next_state"] = df.groupby(["ppt_id", "task_id"])["high_level_code"].shift(-1)
df["valid_transition"] = df["high_level_code"].notna() & df["next_state"].notna()
transitions = df[df["valid_transition"]].copy()
transitions["transition"] = transitions["high_level_code"] + " → " + transitions["next_state"]

# === Global Transition Frequencies and Overall Gini ===
overall = transitions["transition"].value_counts().reset_index()
overall.columns = ["transition", "count"]
overall["proportion"] = overall["count"] / overall["count"].sum()
overall_gini = gini_coefficient(overall["count"].values)

# Save transition counts with Gini
overall["gini"] = ""
summary_row = pd.DataFrame([{
    "transition": "ALL_TRANSITIONS",
    "count": "",
    "proportion": "",
    "gini": overall_gini
}])
combined = pd.concat([overall, summary_row], ignore_index=True)
combined.to_csv(OUTPUT_DIR / "transitions_with_overall_gini.csv", index=False)

# Save Gini stats
gini_stats = {
    "overall_gini": overall_gini,
    "mean_transition_count": overall["count"].mean(),
    "std_transition_count": overall["count"].std(),
    "skew_transition_count": skew(overall["count"])
}
pd.DataFrame([gini_stats]).to_csv(OUTPUT_DIR / "gini_summary_stats.csv", index=False)

# === Gini by Participant ===
ppt_transitions = transitions.groupby(["ppt_id", "transition"]).size().unstack(fill_value=0)
ppt_gini = ppt_transitions.apply(gini_coefficient, axis=1).reset_index()
ppt_gini.columns = ["ppt_id", "gini_coefficient"]

def assign_gini_bin(g):
    if g < 0.3:
        return "Balanced"
    elif g < 0.6:
        return "Moderate"
    else:
        return "Unbalanced"

# Assign bin
ppt_gini["gini_bin"] = ppt_gini["gini_coefficient"].apply(assign_gini_bin)
# Sort by Gini coefficient descending
ppt_gini_sorted = ppt_gini.sort_values(by="gini_coefficient", ascending=False)
# Save the sorted result
ppt_gini_sorted.to_csv(OUTPUT_DIR / "transition_gini_scores.csv", index=False)


# Merge bin info back
transitions = transitions.merge(ppt_gini[["ppt_id", "gini_bin"]], on="ppt_id", how="left")

# Top transitions by bin
gini_trans = transitions.groupby(["gini_bin", "transition"]).size().reset_index(name="count")
bin_total = gini_trans.groupby("gini_bin")["count"].transform("sum")
gini_trans["proportion"] = gini_trans["count"] / bin_total
top_bin_trans = gini_trans[gini_trans["proportion"] > 0.05].copy()
# Sort bin order
bin_order = ["Balanced", "Moderate", "Unbalanced"]
top_bin_trans["gini_bin"] = pd.Categorical(top_bin_trans["gini_bin"], categories=bin_order, ordered=True)
top_bin_trans = top_bin_trans.sort_values(["gini_bin", "proportion"], ascending=[True, False])

top_bin_trans.to_csv(OUTPUT_DIR / "top_transitional_state_by_gini_bin.csv", index=False)

# Barplot by bin
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=top_bin_trans, x="proportion", y="transition", hue="gini_bin", dodge=True)

# Annotate each bar with percentage label (but keep x-axis as raw proportion)
for container in ax.containers:
    labels = [f"{v.get_width() * 100:.1f}%" for v in container]
    ax.bar_label(container, labels=labels, label_type='edge', fontsize=9, padding=3)

plt.title("Top Thought-Pairs by Gini Bin (> 5%) - LLM")
plt.suptitle("Gini bin: Balanced (<0.3), Moderate (0.3–0.6), Unbalanced (≥0.6). Showing transitions >5%.", fontsize=9, y=1.02)
plt.xlabel("Proportion")
plt.ylabel("")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "barplot_top_transitional_state_by_gini_bin.png")
plt.close()



