import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==== Paths ====
HUMAN_PATH = "../analysis_data/full_human_data.csv"
LLM_PATH = "../analysis_data/full_llm_data_subset42.csv"
OUTPUT_DIR = Path("../analysis_results/compare/taskwise_evolution")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== Load Data ====
human_df = pd.read_csv(HUMAN_PATH)
llm_df = pd.read_csv(LLM_PATH)

human_df['group'] = 'Human'
llm_df['group'] = 'LLM'

df = pd.concat([human_df, llm_df], ignore_index=True)

# ==== Summary Table: Accuracy ====
acc_summary = (
    df.groupby(['group', 'task_id'])['accuracy']
    .mean()
    .reset_index()
    .pivot(index='task_id', columns='group', values='accuracy')
    .reset_index()
)
acc_summary['task_id'] = acc_summary['task_id'].astype(int)
acc_summary = acc_summary.sort_values('task_id')

# Add overall row
overall = (
    df.groupby('group')['accuracy'].mean().to_frame().T
)
overall.insert(0, 'task_id', 'Overall')
summary_full = pd.concat([acc_summary, overall], ignore_index=True)
summary_full = summary_full.round(4)
summary_full.to_csv(OUTPUT_DIR / "accuracy_summary_by_task.csv", index=False)

# ==== Plot 1: Accuracy Lineplot ====
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x='task_id', y='accuracy', hue='group', estimator='mean', errorbar=None)
plt.xticks([1, 2, 3, 4, 5])
plt.title("Accuracy Evolution Across Tasks")
plt.xlabel("Task")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_evolution_lineplot.png")
plt.close()

# ==== Plot 2: Cognitive State Frequency by Task ====
state_counts = (
    df.groupby(['group', 'task_id', 'high_level_code'])
    .size()
    .reset_index(name='count')
)

# Normalize within group-task
state_props = state_counts.groupby(['group', 'task_id'])['count'].transform('sum')
state_counts['proportion'] = state_counts['count'] / state_props

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=state_counts,
    x='task_id', y='proportion', hue='high_level_code',
    style='group', markers=True
)
plt.xticks([1, 2, 3, 4, 5])
plt.title("High-Level State Proportion Across Tasks")
plt.xlabel("Task")
plt.ylabel("Proportion")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "state_proportion_evolution_lineplot.png")
plt.close()

# ==== Plot 3: Step Count Evolution ====
step_counts = (
    df.groupby(['group', 'ppt_id', 'task_id'])
    .size()
    .reset_index(name='step_count')
)

plt.figure(figsize=(8, 5))
sns.lineplot(data=step_counts, x='task_id', y='step_count', hue='group', estimator='mean', errorbar='se')
plt.xticks([1, 2, 3, 4, 5])
plt.title("Step Count Evolution Across Tasks")
plt.xlabel("Task")
plt.ylabel("Average Step Count")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "step_count_evolution_lineplot.png")
plt.close()
