import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Set paths ---
DATA_PATH = Path("../analysis_data/full_human_data.csv")
OUTPUT_DIR = Path("../analysis_results/human/freq_proportion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_OUTPUT = OUTPUT_DIR / "human_high_code_frequency_summary.csv"

# --- Load data ---
df = pd.read_csv(DATA_PATH, encoding="utf-8", encoding_errors="replace")

# --- Compute overall frequency + proportion ---
freq = df["high_level_code"].value_counts().reset_index()
freq.columns = ["high_level_code", "count"]
freq["proportion"] = freq["count"] / freq["count"].sum()
freq.to_csv(TABLE_OUTPUT, index=False)
print(f"✅ Saved summary table to {TABLE_OUTPUT}")

# --- Barplot: Frequency of high-level codes ---
plt.figure(figsize=(10, 6))
bar = sns.barplot(
    data=freq,
    y="high_level_code",
    x="count",
    palette="Set2",  # matches LLM script
    order=freq["high_level_code"]
)
for i, row in freq.iterrows():
    bar.text(row["count"] + 5, i, f"{row['proportion']:.1%}", va='center', fontsize=10)
plt.title("High-Level Cognitive State Usage (All Tasks) - Human", fontsize=14)
plt.xlabel("Total Count")
plt.ylabel("High-Level Code")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "human_high_code_barplot_overall.png")
plt.close()
print("✅ Saved barplot: human_high_code_barplot_overall.png")

# --- Task-by-task proportions ---
task_freq = df.groupby(["task_id", "high_level_code"]).size().reset_index(name="count")
task_totals = df.groupby("task_id").size().reset_index(name="total")
task_freq = task_freq.merge(task_totals, on="task_id")
task_freq["proportion"] = task_freq["count"] / task_freq["total"]

# Ensure task_id is treated as categorical for clean axis labels
task_freq["task_id"] = task_freq["task_id"].astype(str)

# --- Lineplot: Proportions of cognitive states across tasks ---
plt.figure(figsize=(11, 7.5))
sns.lineplot(
    data=task_freq,
    x="task_id",
    y="proportion",
    hue="high_level_code",
    marker="o",
    palette="tab10",
    linewidth=2.5
)

# Sort legend to match top-down order of mean proportions
mean_props = task_freq.groupby("high_level_code")["proportion"].mean().sort_values(ascending=False)
handles, labels = plt.gca().get_legend_handles_labels()
ordered_labels = list(mean_props.index)
ordered_handles = [handles[labels.index(lab)] for lab in ordered_labels if lab in labels]
plt.legend(ordered_handles, ordered_labels, title="High-Level Code", bbox_to_anchor=(1.02, 1), loc='upper left')

# Adjust Y-axis to show full range
plt.ylim(0, 0.4)
plt.yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
plt.title("Temporal Evolution of High-Level Cognitive States by Task – Human", fontsize=14)
plt.xlabel("Task ID")
plt.ylabel("Proportion")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "human_high_code_lineplot_by_task.png")
plt.close()
print("✅ Saved lineplot: human_high_code_lineplot_by_task.png")

# --- Grouped Barplot: per task, unstacked ---
pivot_counts = task_freq.pivot(index="task_id", columns="high_level_code", values="proportion").fillna(0)
pivot_counts.plot(
    kind='bar',
    figsize=(12, 6),
    colormap="Set2",
    edgecolor='black',
    width=0.85
)
plt.xticks(rotation=0)
plt.title("Cognitive State Proportions per Task – Human", fontsize=14)
plt.xlabel("Task ID")
plt.ylabel("Proportion")
plt.legend(title="High-Level Code", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "human_grouped_barplot_by_task.png")
plt.close()
print("✅ Saved grouped barplot: human_grouped_barplot_by_task.png")

# --- Stacked Area Chart: Strategy composition per task ---
pivot_props = task_freq.pivot(index="task_id", columns="high_level_code", values="proportion").fillna(0)
pivot_props.plot(
    kind="area",
    stacked=True,
    figsize=(12, 6),
    colormap="tab20c",
    alpha=0.85
)
plt.title("Cognitive Strategy Composition Across Tasks (Stacked Proportions) – Human", fontsize=14)
plt.xlabel("Task ID")
plt.ylabel("Proportion")
plt.legend(title="High-Level Code", loc="upper right", bbox_to_anchor=(0.997, 0.998))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "human_stacked_area_by_task.png")
plt.close()
print("✅ Saved stacked area chart: human_stacked_area_by_task.png")
