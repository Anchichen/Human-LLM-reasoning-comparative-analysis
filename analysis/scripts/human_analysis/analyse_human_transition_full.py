import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from pathlib import Path

# === Config Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "analysis_data" / "full_human_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis_results" / "human" / "transition_full"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Load and Preprocess ===
df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"high-level code": "high_level_code"})
df = df.sort_values(by=["ppt_id", "task_id", "thought_id"])
df["next_state"] = df.groupby(["ppt_id", "task_id"])["high_level_code"].shift(-1)
df["valid_transition"] = df["high_level_code"].notna() & df["next_state"].notna()
transitions = df[df["valid_transition"]].copy()
transitions["transition"] = transitions["high_level_code"] + " → " + transitions["next_state"]

# === Overall Frequency & Proportion ===
overall = transitions["transition"].value_counts().reset_index()
overall.columns = ["transition", "count"]
overall["proportion"] = overall["count"] / overall["count"].sum()
overall["salient"] = overall["proportion"] > 0.05
overall.to_csv(OUTPUT_DIR / "overall_transitional_state_freq.csv", index=False)

# === Barplot: Overall transitions > 5% only ===
top_overall = overall[overall["proportion"] > 0.05]
plt.figure(figsize=(10, 6))
sns.barplot(data=top_overall, x="proportion", y="transition", palette="viridis")
for i, row in top_overall.iterrows():
    plt.text(row["proportion"] + 0.001, i, f'{row["proportion"]:.2%}', va='center')
plt.xlabel("Proportion")
plt.ylabel("")
plt.title("Most Frequent Transition Pairs (> 5%) - Human")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "overall_transitional_state_freq_barplot.png")
plt.close()

# === Accuracy Comparison Table with Z/P/Diff ===
grouped = transitions.groupby(["transition", "accuracy"]).size().unstack(fill_value=0)
grouped.columns = ["Incorrect", "Correct"]
grouped["Total"] = grouped["Correct"] + grouped["Incorrect"]
grouped["Correct_prop"] = grouped["Correct"] / grouped["Total"]
grouped["Incorrect_prop"] = grouped["Incorrect"] / grouped["Total"]
grouped["prop_diff"] = grouped["Correct_prop"] - grouped["Incorrect_prop"]

p1 = grouped["Correct"] / grouped["Total"]
p2 = grouped["Incorrect"] / grouped["Total"]
p = (grouped["Correct"] + grouped["Incorrect"]) / (2 * grouped["Total"])
se = np.sqrt(p * (1 - p) * 2 / grouped["Total"])
grouped["z_score"] = (p1 - p2) / se
grouped["p_value"] = 2 * (1 - norm.cdf(np.abs(grouped["z_score"])))

# Flag significance
grouped["sig_level"] = np.where(grouped["p_value"] < 0.01, "p<0.01",
                                np.where(grouped["p_value"] < 0.05, "p<0.05", ""))
grouped["salient_diff"] = (np.abs(grouped["prop_diff"]) > 0.1) & (grouped["p_value"] < 0.01)

grouped = grouped.reset_index()
grouped.to_csv(OUTPUT_DIR / "transition_comparison_correct_vs_incorrect.csv", index=False)

# === Extract Significant and Salient Transitions ===
significant = grouped[grouped["p_value"] < 0.05]
significant.to_csv(OUTPUT_DIR / "significant_transition_difference.csv", index=False)

salient = grouped[(grouped["p_value"] < 0.01) & (np.abs(grouped["prop_diff"]) > 0.2)]
salient.to_csv(OUTPUT_DIR / "salient_transition_difference.csv", index=False)

# === Barplot: Salient Transitions Only ===
plt.figure(figsize=(10, 6))

# Ensure group labels are correctly formatted as "Correct" and "Incorrect"
sal_melt = salient.melt(
    id_vars="transition",
    value_vars=["Correct_prop", "Incorrect_prop"],
    var_name="Group",
    value_name="Proportion"
)
sal_melt["Group"] = sal_melt["Group"].map({
    "Correct_prop": "Correct",
    "Incorrect_prop": "Incorrect"
})

# Plot
sns.barplot(data=sal_melt, x="Proportion", y="transition", hue="Group", palette="Set2")

# Annotate with percentage on bars
for i, row in sal_melt.iterrows():
    plt.text(row["Proportion"] + 0.005, i % len(salient), f'{row["Proportion"]:.2%}', va='center')

plt.xlabel("Proportion")
plt.ylabel("")
plt.title("Salient Thought-Pair Differences (p<0.01 & >20%) – Human")
plt.legend(title=None)  # ✅ no legend title
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "salient_transitional_state_difference_barplot.png")
plt.close()

print("✅ All refined outputs saved in:", OUTPUT_DIR)

