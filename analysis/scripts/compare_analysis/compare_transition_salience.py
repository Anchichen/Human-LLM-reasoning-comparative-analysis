# compare_transition_salience.py (revised for clean barplot labels and sorting)

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from pathlib import Path

# === PATH SETUP ===
INPUT_HUMAN = Path("../analysis_results/compare/transition_matrices/human_overall_transition_count.csv")
INPUT_LLM = Path("../analysis_results/compare/transition_matrices/llm_overall_transition_count.csv")
OUTPUT_DIR = Path("../analysis_results/compare/transition_salience")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD TRANSITION COUNT MATRICES ===
count_human = pd.read_csv(INPUT_HUMAN, index_col=0)
count_llm = pd.read_csv(INPUT_LLM, index_col=0)

# Align the matrices
all_states = sorted(set(count_human.index).union(set(count_llm.index)))
count_human = count_human.reindex(index=all_states, columns=all_states, fill_value=0)
count_llm = count_llm.reindex(index=all_states, columns=all_states, fill_value=0)

# === CALCULATE TOTAL TRANSITIONS ===
total_h = count_human.values.sum()
total_l = count_llm.values.sum()

# === GATHER ALL TRANSITIONS INTO ROWS ===
data = []
for source in all_states:
    for target in all_states:
        count_h = count_human.loc[source, target]
        count_l = count_llm.loc[source, target]
        prop_h = count_h / total_h
        prop_l = count_l / total_l

        pooled_prop = (count_h + count_l) / (total_h + total_l)
        pooled_std = np.sqrt(pooled_prop * (1 - pooled_prop) * (1/total_h + 1/total_l))

        z = (prop_h - prop_l) / pooled_std if pooled_std > 0 else 0
        p = 2 * (1 - norm.cdf(abs(z)))
        diff = prop_h - prop_l
        sig = (p < 0.01) and (abs(diff) >= 0.05)

        data.append({
            "source": source,
            "target": target,
            "human_count": count_h,
            "llm_count": count_l,
            "human_prop": prop_h,
            "llm_prop": prop_l,
            "prop_diff": diff,
            "z_score": z,
            "p_value": p,
            "significant": sig,
            "direction": "Human > LLM" if prop_h > prop_l else "LLM > Human" if prop_l > prop_h else "Equal"
        })

# === TO DATAFRAME AND SAVE ===
df = pd.DataFrame(data)
df_sorted = df.sort_values(by=["significant", "p_value", "prop_diff"], ascending=[False, True, False])

def format_pval(p):
    return f"**{p:.4f}**" if p < 0.01 else f"*{p:.4f}*" if p < 0.05 else f"{p:.4f}"

def format_diff(d):
    return f"**{d:.2f}**" if abs(d) >= 0.05 else f"{d:.2f}"

df_sorted["p_value"] = df_sorted["p_value"].apply(format_pval)
df_sorted["prop_diff"] = df_sorted["prop_diff"].apply(format_diff)
df_sorted.to_csv(OUTPUT_DIR / "all_transitions_comparison.csv", index=False)

# === SAVE FILTERED SIGNIFICANT SETS ===
p_001_dif_005 = df[(df["p_value"] < 0.01) & (df["prop_diff"].abs() >= 0.05)]
p_under_001 = df[df["p_value"] < 0.01].sort_values("prop_diff", ascending=False)

p_001_dif_005.to_csv(OUTPUT_DIR / "salient_transition_p0.01_diff_0.05.csv", index=False)
p_under_001.to_csv(OUTPUT_DIR / "transition_salience_p_under_0.01.csv", index=False)


# === HELPER: Format label (two-line, aligned left inside, centered on bar) ===
def format_label(source, target):
    return f"{source}→\n {target}"

def add_percentage_labels(ax, x, values, offset=0.002, width_shift=0.0):
    for i, val in enumerate(values):
        ax.text(
            x[i] + width_shift,
            val + offset,
            f"{val*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
        )

# === BARPLOT OF TRANSITION PROPORTIONS WITH SIGNIFICANCE MARKERS ===
plot_df = df.copy()
plot_df["mean_prop"] = plot_df[["human_prop", "llm_prop"]].mean(axis=1)
plot_df = plot_df[plot_df["mean_prop"] > 0.05]

# === Rename long state names early ===
rename_map = {
    "ProcessingScope": "Processing",
    "Evaluation/Monitoring": "Evaluation"
}
df["source"] = df["source"].replace(rename_map)
df["target"] = df["target"].replace(rename_map)

# Identify which transitions are significant
sig_set = set(
    df[(df["p_value"] < 0.01) & (df["prop_diff"].abs() >= 0.05)]
    .apply(lambda row: (row["source"], row["target"]), axis=1)
)

plot_df["is_significant"] = plot_df.apply(
    lambda row: (row["source"], row["target"]) in sig_set, axis=1
)

plot_df = plot_df.sort_values("llm_prop", ascending=True)

def format_label_arrow_break(source, target, is_significant=False):
    label = f"{source} →\n{target}"
    return label + "**" if is_significant else label

x_labels = plot_df.apply(
    lambda row: format_label_arrow_break(row["source"], row["target"], row["is_significant"]),
    axis=1
)

x = np.arange(len(x_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x - width/2, plot_df["human_prop"], width, label="Human", color="skyblue")
ax.bar(x + width/2, plot_df["llm_prop"], width, label="LLM", color="lightcoral")
ax.set_xticks(x)
ax.set_xticklabels(x_labels, ha="center", fontsize=10)

def add_percentage_labels(ax, x, values, offset=0.002, width_shift=0.0):
    for i, val in enumerate(values):
        ax.text(
            x[i] + width_shift,
            val + offset,
            f"{val*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

add_percentage_labels(ax, x, plot_df["human_prop"], width_shift=-width/2)
add_percentage_labels(ax, x, plot_df["llm_prop"], width_shift=width/2)

ax.set_ylabel("Proportion")
ax.set_title("Transition Proportion Comparison (Human vs LLM)")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "barplot_transition_proportion_comparison.png", dpi=300)
plt.close()



# === HUMAN ONLY PLOT ===
df_h = df[df["human_prop"] > 0.05].copy()
df_h = df_h.sort_values("human_prop", ascending=True)
labels_h = df_h.apply(lambda row: format_label(row['source'], row['target']), axis=1)

x = np.arange(len(labels_h))
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(x, df_h["human_prop"], color='skyblue')
ax.set_xticks(x)
ax.set_xticklabels(labels_h, ha="center", fontsize=10)

add_percentage_labels(ax, x, df_h["human_prop"])

ax.set_ylabel("Proportion")
ax.set_title("Human Transition Distribution (Proportion > 0.05)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hist_human_transition_prop_gt_0.05.png", dpi=300)
plt.close()


# === LLM ONLY PLOT ===
df_l = df[df["llm_prop"] > 0.05].copy()
df_l = df_l.sort_values("llm_prop", ascending=True)
labels_l = df_l.apply(lambda row: format_label(row['source'], row['target']), axis=1)

x = np.arange(len(labels_l))
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(x, df_l["llm_prop"], color='lightcoral')
ax.set_xticks(x)
ax.set_xticklabels(labels_l, ha="center", fontsize=9)

add_percentage_labels(ax, x, df_l["llm_prop"])

ax.set_ylabel("Proportion")
ax.set_title("LLM Transition Distribution (Proportion > 0.05)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hist_llm_transition_prop_gt_0.05.png", dpi=300)
plt.close()


# === BARPLOT TRANSITION DISTRIBUTION COMPARISON WITH SIGNIFICANCE MARKERS ===
df_both = df[(df["human_prop"] > 0.05) | (df["llm_prop"] > 0.05)].copy()


df_both["is_significant"] = df_both.apply(
    lambda row: (row["source"], row["target"]) in sig_set, axis=1
)

df_both = df_both.sort_values("llm_prop", ascending=True)

x_labels = df_both.apply(
    lambda row: format_label_arrow_break(row["source"], row["target"], row["is_significant"]),
    axis=1
)

x = np.arange(len(x_labels))

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width/2, df_both["human_prop"], width, label="Human", color="skyblue")
ax.bar(x + width/2, df_both["llm_prop"], width, label="LLM", color="lightcoral")
ax.set_xticks(x)
ax.set_xticklabels(x_labels, ha="center", fontsize=15)

add_percentage_labels(ax, x, df_both["human_prop"], width_shift=-width/2)
add_percentage_labels(ax, x, df_both["llm_prop"], width_shift=width/2)

ax.set_ylabel("Proportion")
ax.set_title("Transition Distribution and Comparison (Human vs LLM)", size = 16, weight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "barplot_transition_distribution_comparison.png", dpi=300)
plt.close()


# === LABEL RECURSIVE VS DIVERGENT ===
df["transition_type"] = np.where(df["source"] == df["target"], "Recursive", "Divergent")

# === AGGREGATE TOTAL PROPORTIONS ===
agg = df.groupby("transition_type")[["human_prop", "llm_prop"]].sum().reset_index()

from scipy.stats import chi2_contingency

# === Build contingency table ===
human_rec = df[(df["transition_type"] == "Recursive")]["human_count"].sum()
human_div = df[(df["transition_type"] == "Divergent")]["human_count"].sum()
llm_rec = df[(df["transition_type"] == "Recursive")]["llm_count"].sum()
llm_div = df[(df["transition_type"] == "Divergent")]["llm_count"].sum()

contingency_table = np.array([
    [human_rec, human_div],
    [llm_rec, llm_div]
])

# === Run Chi-square test ===
chi2, p_val, dof, expected = chi2_contingency(contingency_table)

print("Chi-square Test Results:")
print(f"  χ² = {chi2:.4f}")
print(f"  p-value = {p_val:.4e}")
print(f"  Degrees of freedom = {dof}")

# === PLOT BARPLOT: Recursive vs Divergent ===
fig, ax = plt.subplots(figsize=(6, 5))
bar_x = np.arange(len(agg))
width = 0.35
ax.bar(bar_x - width/2, agg["human_prop"], width, label="Human", color="skyblue")
ax.bar(bar_x + width/2, agg["llm_prop"], width, label="LLM", color="lightcoral")
ax.set_xticks(bar_x)
ax.set_xticklabels(agg["transition_type"], fontsize=13)
ax.set_ylabel("Total Proportion", fontsize=13)
ax.set_title("Recursive vs Divergent Transition Usage", fontsize=14, weight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "barplot_recursive_vs_divergent_comparison.png", dpi=300)
plt.close()

# === SAVE PROPORTIONS ===
agg.to_csv(OUTPUT_DIR / "recursive_vs_divergent_proportions.csv", index=False)

