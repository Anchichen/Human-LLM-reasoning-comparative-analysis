# human Source-State Transition Comparison: Target Shift + Gini Diff + Significance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import norm

# --- Paths ---
DATA_PATH = Path("../analysis_data/full_human_data.csv")
OUT_DIR = Path("../analysis_results/human/transition_source_state")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load ---
df = pd.read_csv(DATA_PATH)
df = df.sort_values(by=["ppt_id", "task_id", "thought_id"])
df["next_code"] = df.groupby(["ppt_id", "task_id"])["high_level_code"].shift(-1)
df_trans = df.dropna(subset=["next_code"]).copy()
df = df.dropna(subset=["next_code"])
df["transition"] = list(zip(df["high_level_code"], df["next_code"]))

# === 1. Target shift difference plots ===
def get_target_dists(group_df):
    return group_df.groupby("high_level_code")["next_code"].value_counts(normalize=True).unstack().fillna(0)

dist_corr = get_target_dists(df[df["accuracy"] == 1])
dist_incorr = get_target_dists(df[df["accuracy"] == 0])

dist_diff = (dist_corr - dist_incorr).fillna(0)
dist_diff.to_csv(OUT_DIR / "source_state_target_distribution_diff.csv")
# Title: Difference in Target Transition Proportions by Source State (Correct - Incorrect)
# Description: Each cell represents the proportion difference for a transition from source (row) to target (column)

shift_long = dist_diff.reset_index().melt(id_vars="high_level_code", var_name="target", value_name="prop_diff")
shift_long = shift_long[shift_long["prop_diff"].abs() > 0.05].copy()
shift_long.rename(columns={"high_level_code": "source"}, inplace=True)

shift_long["direction"] = shift_long["prop_diff"].apply(
    lambda x: "More in Correct" if x > 0 else "More in Incorrect"
)

g = sns.FacetGrid(shift_long, col="source", col_wrap=4, sharex=False, height=3.5)
g.map_dataframe(
    sns.barplot,
    x="prop_diff", y="target",
    hue="direction", dodge=False,
    palette={"More in Correct": "salmon", "More in Incorrect": "steelblue"}
)
g.set_titles("{col_name}")
g.set_axis_labels("", "")
g.fig.supxlabel("Proportion Difference (Correct – Incorrect)", fontsize=11.5)
for ax in g.axes.flat:
    ax.axvline(0, color="gray", linestyle="--")
g.add_legend(title=None, loc='upper right', bbox_to_anchor=(0.95, 0.97), borderaxespad=0, frameon=True)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Target Shift by Source State - Human", fontsize=14)
g.savefig(OUT_DIR / "facet_target_shift_diff.png", bbox_inches="tight")
plt.close()

# === 2. Gini difference by source ===
def gini(array):
    array = np.sort(np.array(array)) + 1e-9
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

# Compute Gini for each source in correct and incorrect
source_gini_corr = dist_corr.apply(gini, axis=1)
source_gini_incorr = dist_incorr.apply(gini, axis=1)

# Combine into DataFrame
source_ginis = pd.DataFrame({
    "Correct Gini": source_gini_corr,
    "Incorrect Gini": source_gini_incorr
})
source_ginis["Gini Difference (Correct - Incorrect)"] = source_ginis["Correct Gini"] - source_ginis["Incorrect Gini"]

# Significance flag and bold display
source_ginis["Significant (|ΔGini| > 0.05)"] = source_ginis["Gini Difference (Correct - Incorrect)"].abs() > 0.05
source_ginis["Gini Difference (Formatted)"] = source_ginis["Gini Difference (Correct - Incorrect)"].apply(
    lambda x: f"**{x:.4f}**" if abs(x) > 0.05 else f"{x:.4f}"
)

# Save table
source_ginis.to_csv(OUT_DIR / "source_state_gini_comparison.csv")

# Plot
plt.figure(figsize=(8,5))
sorted_diff = source_ginis.sort_values("Gini Difference (Correct - Incorrect)")
sns.barplot(
    data=sorted_diff.reset_index(),
    x="Gini Difference (Correct - Incorrect)",
    y="high_level_code",
    hue="high_level_code",
    palette="vlag",
    legend=False
)
plt.axvline(0, color="gray", linestyle="--")
plt.title("Gini Difference of Transition Spread per Source State - Human")
plt.xlabel("Δ Gini (Correct - Incorrect)")
plt.ylabel("Source State")
plt.tight_layout()
plt.savefig(OUT_DIR / "barplot_gini_difference_by_source.png")
plt.close()


# === 3. Source-State Transition Significance ===
def prop_z_test(p1, n1, p2, n2):
    p_comb = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p_comb * (1 - p_comb) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0, 1
    z = (p1 - p2) / se
    p_val = 2 * (1 - norm.cdf(abs(z)))
    return z, p_val

df_corr = df[df["accuracy"] == 1]
df_incorr = df[df["accuracy"] == 0]
count_corr = df_corr["transition"].value_counts()
count_incorr = df_incorr["transition"].value_counts()
n1 = count_corr.sum()
n2 = count_incorr.sum()
all_transitions = set(count_corr.index).union(count_incorr.index)

rows = []
for trans in all_transitions:
    from_state, to_state = trans
    c1 = count_corr.get(trans, 0)
    c2 = count_incorr.get(trans, 0)
    p1 = c1 / n1
    p2 = c2 / n2
    z, p = prop_z_test(p1, n1, p2, n2)
    prop_diff = p1 - p2  # use raw proportion, not percentage
    sig_flag = (p < 0.05) and (abs(prop_diff) > 0.015)  # threshold at 1.5% = 0.015
    direction = "correct > incorrect" if prop_diff > 0 else "incorrect > correct"

    row = {
        "From": from_state,
        "To": to_state,
        "Correct Proportion": round(p1, 4),
        "Incorrect Proportion": round(p2, 4),
        "Proportion Difference (Correct - Incorrect)": f"**{prop_diff:.4f}**" if abs(prop_diff) > 0.015 else f"{prop_diff:.4f}",
        "Z-score": round(z, 2),
        "p-value": f"**{p:.4f}**" if p < 0.05 else f"{p:.4f}",
        "Significant": sig_flag,
        "Direction": direction
    }
    rows.append(row)


sig_df = pd.DataFrame(rows)
sig_df.sort_values(by=["Significant", "Correct Proportion"], ascending=[False, False], inplace=True)
sig_df.to_csv(OUT_DIR / "source_state_transition_comparison.csv", index=False)

sig_df[sig_df["Significant"] == True].to_csv(
    OUT_DIR / "significant_source_state_transition.csv", index=False)

salient = sig_df[sig_df["Significant"] == True].copy()
salient["label"] = salient["From"] + " → " + salient["To"]

# === SWITCHED color palette ===
palette = {
    "correct > incorrect": "salmon",
    "incorrect > correct": "steelblue"
}

plt.figure(figsize=(10, 6))
sns.barplot(data=salient, x="Z-score", y="label", hue="Direction", dodge=False, palette=palette)
plt.axvline(0, color="gray", linestyle="--")
plt.title("Significant Transition Differences by Source State – Human")
plt.xlabel("Z-score")
plt.ylabel("")
plt.legend(title=None, loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig(OUT_DIR / "barplot_significant_transition_diff.png")
plt.close()

