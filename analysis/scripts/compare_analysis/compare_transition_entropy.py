# transition_entropy_analysis.py
# Compute and compare transition entropy for Human and LLM participants

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import pairwise
from collections import Counter
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
from statsmodels.formula.api import logit

# === SET FONT STYLE ===
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.titlesize'] = 16   # Title font size
plt.rcParams['axes.labelsize'] = 14   # X and Y axis label size
plt.rcParams['xtick.labelsize'] = 12  # X tick label size
plt.rcParams['ytick.labelsize'] = 12  # Y tick label size
plt.rcParams['legend.fontsize'] = 12  # Legend font size

# === PATHS ===
DATA_HUMAN = Path("../analysis_data/full_human_data.csv")
DATA_LLM = Path("../analysis_data/full_llm_data_subset42.csv")
OUTPUT_DIR = Path("../analysis_results/compare/transition_entropy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD DATA ===
def load_data(path, group_label):
    df = pd.read_csv(path)
    df = df[['task_id', 'ppt_id', 'high_level_code', 'accuracy']]
    df['group'] = group_label
    return df

human = load_data(DATA_HUMAN, 'Human')
llm = load_data(DATA_LLM, 'LLM')
data = pd.concat([human, llm], ignore_index=True)

# === HELPER: COMPUTE ENTROPY ===
def compute_entropy(transition_counts):
    total = sum(transition_counts.values())
    probs = [count / total for count in transition_counts.values() if count > 0]
    return -sum(p * np.log2(p) for p in probs) if probs else 0.0

def compute_entropy_by_group(data, allowed_transitions=None):
    entropy_rows = []
    for (group, ppt_id), df_group in data.groupby(['group', 'ppt_id']):
        transitions = []
        for _, df_task in df_group.groupby('task_id'):
            codes = df_task['high_level_code'].tolist()
            transitions += list(pairwise(codes))

        if allowed_transitions is not None:
            transitions = [t for t in transitions if t in allowed_transitions]

        counter = Counter(transitions)
        entropy = compute_entropy(counter)
        acc = df_group['accuracy'].iloc[0]
        entropy_rows.append((group, ppt_id, entropy, acc))
    return pd.DataFrame(entropy_rows, columns=['group', 'ppt_id', 'entropy', 'accuracy'])

# === STEP 1: Entropy using all transitions ===
entropy_full = compute_entropy_by_group(data)
entropy_full['source'] = 'Full'

# === STEP 2: Entropy using only LLM-used transitions ===
llm_transitions = []
for _, df_group in llm.groupby(['ppt_id']):
    for _, df_task in df_group.groupby('task_id'):
        codes = df_task['high_level_code'].tolist()
        llm_transitions += list(pairwise(codes))
allowed_transitions = set(llm_transitions)

entropy_llmonly = compute_entropy_by_group(data, allowed_transitions=allowed_transitions)
entropy_llmonly['source'] = 'LLM-Only'

# === COMBINE BOTH ===
entropy_combined = pd.concat([entropy_full, entropy_llmonly], ignore_index=True)

# === SAVE ENTROPY TABLES ===
entropy_combined.to_csv(OUTPUT_DIR / "entropy_by_participant_combined.csv", index=False)

# === SUMMARY TABLE + MANN-WHITNEY TEST (for both sources) ===
summary_rows = []
for source_type in ['Full', 'LLM-Only']:
    df = entropy_combined[entropy_combined['source'] == source_type]
    stats = df.groupby('group')['entropy'].agg(['mean', 'std', 'median', 'min', 'max', 'count']).reset_index()
    stats['source'] = source_type

    # Mann-Whitney test
    human_vals = df[df['group'] == 'Human']['entropy']
    llm_vals = df[df['group'] == 'LLM']['entropy']
    u_stat, p_val = mannwhitneyu(human_vals, llm_vals, alternative='two-sided')
    stats['mannwhitney_U'] = u_stat
    stats['p_value'] = p_val

    summary_rows.append(stats)

summary_df = pd.concat(summary_rows, ignore_index=True)
summary_df.to_csv(OUTPUT_DIR / "entropy_summary_with_tests.csv", index=False)

# === BOXPLOTS ===
for src in ['Full', 'LLM-Only']:
    df = entropy_combined[entropy_combined['source'] == src]
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='group', y='entropy', palette='pastel')
    sns.stripplot(data=df, x='group', y='entropy', color='black', alpha=0.6, jitter=True)
    plt.title(f"Transition Entropy per Participant/Run ({src} cognitive states)")
    plt.ylabel("Entropy (bits)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"boxplot_entropy_by_group_{src.lower().replace('-', '')}.png", dpi=300)
    plt.close()

# === STEP 3: Logistic Regression (Entropy → Accuracy) for Human and LLM ===
for group in ['Human', 'LLM']:
    df = entropy_full[entropy_full['group'] == group].copy()
    df['accuracy'] = df['accuracy'].astype(int)
    model = logit("accuracy ~ entropy", data=df).fit(disp=0)

    with open(OUTPUT_DIR / f"regression_entropy_accuracy_{group.lower()}.txt", "w") as f:
        f.write(f"Logistic Regression: Entropy → Accuracy ({group})\n\n")
        f.write(model.summary().as_text())
        f.write("\nInterpretation:\n")
        f.write("Positive coefficient → higher entropy increases chance of correct prediction.\n")

    # Boxplot of entropy vs accuracy
    plt.figure(figsize=(6, 5))
    sns.boxplot(x='accuracy', y='entropy', data=df, palette='Set2')
    sns.stripplot(x='accuracy', y='entropy', data=df, color='black', jitter=True, alpha=0.6)
    plt.title(f"{group} Entropy vs Accuracy")
    plt.xlabel("Accuracy (0 = incorrect, 1 = correct)")
    plt.ylabel("Entropy (bits)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"entropy_vs_accuracy_{group.lower()}.png", dpi=300)
    plt.close()


# === INTEGRATED PLOT: Entropy vs Accuracy (Human vs LLM) ===
# BIN entropy into quantiles and compute mean accuracy
df_plot = entropy_full.copy()
df_plot['accuracy'] = df_plot['accuracy'].astype(int)
df_plot['entropy_bin'] = pd.qcut(df_plot['entropy'], q=6, duplicates='drop').astype(str)
grouped = df_plot.groupby(['entropy_bin', 'group']).agg(mean_acc=('accuracy', 'mean')).reset_index()

plt.figure(figsize=(8, 5))
sns.lineplot(data=grouped, x='entropy_bin', y='mean_acc', hue='group', marker='o')
plt.xticks(rotation=45)
plt.title("Binned Entropy vs Accuracy")
plt.ylabel("Mean Accuracy")
plt.xlabel("Entropy Bin")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "entropy_vs_accuracy_binned.png", dpi=300)
plt.close()



print("✅ Transition entropy comparison and regression completed.")
