# LLM Transition Frequency, Salience, and Group Difference Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency
from collections import Counter
from itertools import product

# --- Paths ---
DATA_PATH = Path("../analysis_data/full_llm_data.csv")
OUT_DIR = Path("../analysis_results/llm/transition_freq")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load and prepare data ---
df = pd.read_csv(DATA_PATH)
df = df.sort_values(by=["ppt_id", "task_id", "thought_id"])
df["next_code"] = df.groupby(["ppt_id", "task_id"])["high_level_code"].shift(-1)
df = df.dropna(subset=["next_code"])
df["transition"] = list(zip(df["high_level_code"], df["next_code"]))

# --- Frequency and proportion of transitions ---
def get_transition_freqs(data):
    count = Counter(data["transition"])
    total = sum(count.values())
    rows = [(a, b, c, c/total) for (a, b), c in count.items()]
    return pd.DataFrame(rows, columns=["source", "target", "count", "proportion"])

# Overall frequency
all_trans = get_transition_freqs(df)
all_trans = all_trans.sort_values("proportion", ascending=False)
all_trans.to_csv(OUT_DIR / "transition_frequency_overall.csv", index=False)


# --- Grouped comparison ---
df_corr = df[df["accuracy"] == 1]
df_incorr = df[df["accuracy"] == 0]
trans_corr = get_transition_freqs(df_corr)
trans_incorr = get_transition_freqs(df_incorr)

# --- Merge for comparison ---
merged = pd.merge(trans_corr, trans_incorr, on=["source", "target"], how="outer", suffixes=("_corr", "_incorr")).fillna(0)
merged["prop_diff"] = merged["proportion_corr"] - merged["proportion_incorr"]

# --- Chi2 test ---
# Needed to define totals before use
df_corr = df[df["accuracy"] == 1]
df_incorr = df[df["accuracy"] == 0]
total_corr = df_corr.shape[0]
total_incorr = df_incorr.shape[0]


def compute_p(row):
    a = row["count_corr"]
    b = total_corr - a
    c = row["count_incorr"]
    d = total_incorr - c
    # Avoid small cell artifacts
    if a + c < 5:
        return 1.0
    obs = [[a, b], [c, d]]
    try:
        _, p, _, _ = chi2_contingency(obs)
    except ValueError:
        p = 1.0
    return p


merged["p"] = merged.apply(compute_p, axis=1)
merged["p"] = merged["p"].round(5)  # ✅ Add this line here

from scipy.stats import norm

# Compute z-score based on pooled standard error
merged["z_score"] = merged["prop_diff"] / (1e-9 + np.sqrt(
    (merged["proportion_corr"] * (1 - merged["proportion_corr"]) / total_corr) +
    (merged["proportion_incorr"] * (1 - merged["proportion_incorr"]) / total_incorr)
))

# Mark significance using combined criteria
# Annotate the significance standard
def get_significance_flag(row):
    if row["p"] < 0.01:
        return "p < 0.01"
    elif row["p"] < 0.05:
        return "p < 0.05"
    else:
        return ""

merged["significance_flag"] = merged.apply(get_significance_flag, axis=1)


merged["p_str"] = merged["p"].apply(lambda x: f"**{x:.5f}**" if x < 0.05 else f"{x:.5f}")
merged["diff_str"] = merged["prop_diff"].apply(lambda x: f"**{x:.3f}**" if abs(x) > 0.1 else f"{x:.3f}")

merged.to_csv(OUT_DIR / "transition_comparison_correct_vs_incorrect.csv", index=False)

# --- Extract salient transitions ---
# Keep all p < 0.05 transitions
salient = merged[merged["p"] < 0.05].copy()
salient["significance_flag"] = salient["p"].apply(lambda x: "p < 0.01" if x < 0.01 else "p < 0.05")

# Sort by p-value first, then by absolute z-score descending
salient = salient.sort_values(by=["p", "z_score"], ascending=[True, False])

salient.to_csv(OUT_DIR / "significant_transition_difference.csv", index=False)



# --- Barplot of salient ---
# Use only p < 0.01 transitions for plotting
salient_plot = merged[merged["p"] < 0.01].copy()
salient_plot = salient_plot.sort_values(by=["p", "z_score"], ascending=[True, False])

plot_df = salient_plot.melt(
    id_vars=["source", "target"],
    value_vars=["proportion_corr", "proportion_incorr"],
    var_name="group",
    value_name="proportion"
)
plot_df["transition"] = plot_df["source"] + " → " + plot_df["target"]
plot_df["group"] = plot_df["group"].map({
    "proportion_corr": "Correct",
    "proportion_incorr": "Incorrect"
})

plt.figure(figsize=(12, 6))
sns.barplot(data=plot_df, x="proportion", y="transition", hue="group")
plt.title("Significant Transition Proportion Differences (p < 0.01) - LLM")
plt.xlabel("Proportion")
plt.ylabel("Transition")
plt.tight_layout()
plt.savefig(OUT_DIR / "barplot_significant_transition_diff_p001_LLM.png")
plt.close()




# Load overall frequencies
overall_freq = pd.read_csv(OUT_DIR / "transition_frequency_overall.csv")
salient_overall = overall_freq[overall_freq["proportion"] > 0.05].copy()
salient_overall["label"] = salient_overall["source"] + " → " + salient_overall["target"]

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=salient_overall.sort_values("proportion", ascending=False),
            y="label", x="proportion", color="steelblue")
plt.title("Top Frequent Thought Transitions (Proportion > 5%) - LLM")
plt.ylabel("")  # Remove y-axis label
plt.tight_layout()
plt.savefig(OUT_DIR / "barplot_salient_transition_overall.png")
plt.close()

# --- Reference Barplot: Proportion Difference by Transition (Correct - Incorrect, p < 0.01) ---

salient_diff = merged[merged["p"] < 0.01].copy()
salient_diff["transition"] = salient_diff["source"] + " → " + salient_diff["target"]
salient_diff = salient_diff.sort_values("prop_diff")

plt.figure(figsize=(10, 6))
sns.barplot(
    data=salient_diff,
    y="transition",
    x="prop_diff",
    hue="target",
    dodge=False,
    palette="tab10",        # Higher contrast than Set2
    edgecolor="black"
)
plt.axvline(0, color="gray", linestyle="--")
plt.title("Salient Transition Differences (Correct vs Incorrect, p < 0.01) - LLM")
plt.xlabel("Proportion Difference (Correct − Incorrect)")
plt.ylabel("Transition")
plt.legend(title="Target State", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(OUT_DIR / "barplot_proportion_difference_directional_LLM.png")
plt.close()
