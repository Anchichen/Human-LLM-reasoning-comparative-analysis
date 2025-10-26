
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np
from pathlib import Path

# === Load full human and LLM data ===
df_human = pd.read_csv("../analysis_data/full_human_data.csv")
df_llm = pd.read_csv("../analysis_data/full_llm_data_subset42.csv")

# === Standardize and Add Group Labels ===
df_human['group'] = 'Human'
df_llm['group'] = 'LLM'
df = pd.concat([df_human, df_llm], ignore_index=True)

# === Drop duplicates: One row per ppt_id × task_id ===
df_final = df.drop_duplicates(subset=['ppt_id', 'task_id'])

# === Overall Accuracy Summary ===
overall_summary = df_final.groupby('group')['accuracy'].agg(['mean', 'std', 'count']).reset_index()
print("=== Overall Accuracy Summary ===")
print(overall_summary)

# === Per-Task Accuracy Summary ===
task_summary = df_final.groupby(['group', 'task_id'])['accuracy'].agg(['mean', 'std', 'count']).reset_index()
print("\n=== Per-Task Accuracy Summary ===")
print(task_summary)

# === Compute mean accuracy per participant ===
df_grouped = df_final.groupby(['group', 'ppt_id']).agg(
    mean_accuracy=('accuracy', 'mean'),
    std_accuracy=('accuracy', 'std')
).reset_index()

# === Split Human vs LLM for t-test ===
human_acc = df_grouped[df_grouped['group'] == 'Human']['mean_accuracy']
llm_acc = df_grouped[df_grouped['group'] == 'LLM']['mean_accuracy']

# === Welch's t-test ===
t_stat, p_val = ttest_ind(human_acc, llm_acc, equal_var=False)

# === Descriptive Stats ===
mean_h, std_h = human_acc.mean(), human_acc.std()
mean_l, std_l = llm_acc.mean(), llm_acc.std()
n_h, n_l = len(human_acc), len(llm_acc)

# === Cohen’s d ===
pooled_sd = np.sqrt((std_h ** 2 + std_l ** 2) / 2)
cohens_d = (mean_h - mean_l) / pooled_sd

# === Report Overall Group Comparison ===
print("\n=== Human vs. LLM Accuracy Comparison (per participant) ===")
print(f"Human: M = {mean_h:.3f}, SD = {std_h:.3f}, n = {n_h}")
print(f"LLM:   M = {mean_l:.3f}, SD = {std_l:.3f}, n = {n_l}")
print(f"t({n_h + n_l - 2:.1f}) = {t_stat:.3f}, p = {p_val:.4f}")
print(f"Cohen's d = {cohens_d:.3f}")

if p_val < 0.05:
    print("=> Result is statistically significant (p < 0.05)")
else:
    print("=> Result is NOT statistically significant")

# === Per-Task Significance Tests ===
print("\n=== Task-wise Significance Tests ===")
task_ids = df_final['task_id'].unique()
for task in sorted(task_ids):
    h = df_final[(df_final['group'] == 'Human') & (df_final['task_id'] == task)]['accuracy']
    l = df_final[(df_final['group'] == 'LLM') & (df_final['task_id'] == task)]['accuracy']
    t, p = ttest_ind(h, l, equal_var=False)
    m_h, s_h = h.mean(), h.std()
    m_l, s_l = l.mean(), l.std()
    print(f"Task {task}: Human M={m_h:.3f}, SD={s_h:.3f} | LLM M={m_l:.3f}, SD={s_l:.3f} | t = {t:.3f}, p = {p:.4f}")


import matplotlib.pyplot as plt
import seaborn as sns

# === Use Arial for all plot text ===
plt.rcParams['font.family'] = 'Arial'

# === Accuracy Line Plot by Task (Styled) ===
plt.figure(figsize=(7, 4))

palette = {'Human': '#4682B4', 'LLM': 'salmon'}

sns.lineplot(
    data=task_summary,
    x='task_id',
    y='mean',
    hue='group',
    palette=palette,
    marker='o',
    linewidth=2.5,
    markersize=8
)

plt.xlabel("Task", fontsize=14)
plt.ylabel("Mean Accuracy", fontsize=14)
plt.title("Accuracy by Task: Human vs LLM", fontsize=16)
plt.xticks([1, 2, 3, 4, 5], fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 1)

plt.legend(title='', loc='upper right', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

plt.savefig("accuracy_by_task_lineplot.png", dpi=300)
plt.show()


