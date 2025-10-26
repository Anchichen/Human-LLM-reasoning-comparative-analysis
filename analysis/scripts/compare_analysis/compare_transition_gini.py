# compare_transition_gini.py
# Compute and compare transition Gini index for Human and LLM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import pairwise
from collections import Counter

# === PATHS ===
DATA_HUMAN = Path("../analysis_data/full_human_data.csv")
DATA_LLM = Path("../analysis_data/full_llm_data_subset42.csv")
OUTPUT_DIR = Path("../analysis_results/compare/transition_gini")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === GINI FUNCTION ===
def gini_coefficient(x):
    if len(x) == 0:
        return 0.0
    x = np.array(x)
    if np.sum(x) == 0:
        return 0.0
    sorted_x = np.sort(x)
    n = len(x)
    cumulative = np.cumsum(sorted_x)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n

# === LOAD DATA ===
def load_data(path, group_label):
    df = pd.read_csv(path)
    df = df[['task_id', 'ppt_id', 'high_level_code']]
    df['group'] = group_label
    return df

human = load_data(DATA_HUMAN, 'Human')
llm = load_data(DATA_LLM, 'LLM')
data = pd.concat([human, llm], ignore_index=True)

# === COMPUTE TRANSITIONS ===
all_transitions = []

for (group, ppt, task), df_group in data.groupby(['group', 'ppt_id', 'task_id']):
    codes = df_group['high_level_code'].tolist()
    transitions = list(pairwise(codes))
    all_transitions.extend([(group, ppt, task, f"{a} → {b}") for a, b in transitions])

transition_df = pd.DataFrame(all_transitions, columns=['group', 'ppt_id', 'task_id', 'transition'])

# === PART 1: OVERALL GINI ===
overall = (
    transition_df.groupby('group')['transition']
    .value_counts()
    .groupby(level=0)
    .apply(lambda x: gini_coefficient(x.values))
    .reset_index(name='overall_gini')
)
overall.to_csv(OUTPUT_DIR / "overall_transition_gini.csv", index=False)

# === PART 2: PER-TRANSITION GINI ===
all_transitions_by_group = (
    transition_df.groupby(['group', 'transition'])
    .size()
    .reset_index(name='count')
)

# Compute per-transition gini across ppt-task blocks
transition_counts = []

for group, df_group in data.groupby('group'):
    counts_by_transition = {}
    for (ppt, task), df_block in df_group.groupby(['ppt_id', 'task_id']):
        codes = df_block['high_level_code'].tolist()
        for a, b in pairwise(codes):
            key = f"{a} → {b}"
            counts_by_transition.setdefault(key, []).append((ppt, task))

    transition_gini = []
    for transition, blocks in counts_by_transition.items():
        counter = Counter(blocks)
        gini = gini_coefficient(list(counter.values()))
        total = sum(counter.values())
        transition_gini.append((group, transition, gini, total))

    df_gini = pd.DataFrame(transition_gini, columns=['group', 'transition', 'gini', 'count'])
    if group == 'Human':
        gini_human = df_gini.copy()
    else:
        gini_llm = df_gini.copy()

# Merge and save
merged = pd.concat([gini_human, gini_llm], ignore_index=True)
merged.to_csv(OUTPUT_DIR / "gini_by_transition.csv", index=False)

# === PART 3: GROUPED GINI ANALYSIS ===
def bin_gini(val):
    if val <= 0.33:
        return 'Low'
    elif val <= 0.66:
        return 'Medium'
    else:
        return 'High'

merged['gini_bin'] = merged['gini'].apply(bin_gini)

# Save top transitions in each bin
top_by_bin = (
    merged.sort_values(['group', 'gini_bin', 'count'], ascending=[True, True, False])
    .groupby(['group', 'gini_bin'])
    .head(10)
)
top_by_bin.to_csv(OUTPUT_DIR / "top_transitions_by_gini_bin.csv", index=False)

# === PART 4: PER-PARTICIPANT TRANSITION GINI ===
ppt_ginis = []

for (group, ppt), df_group in transition_df.groupby(['group', 'ppt_id']):
    counter = Counter(df_group['transition'])
    gini = gini_coefficient(list(counter.values()))
    ppt_ginis.append((group, ppt, gini))

ppt_df = pd.DataFrame(ppt_ginis, columns=['group', 'ppt_id', 'gini'])

# Sort properly by numeric part of ppt_id
ppt_df['ppt_num'] = ppt_df['ppt_id'].str.extract(r'(\d+)').astype(int)
ppt_df = ppt_df.sort_values(by=['group', 'ppt_num'])
ppt_df.drop(columns='ppt_num', inplace=True)
ppt_df.to_csv(OUTPUT_DIR / "transition_gini_by_participant.csv", index=False)

# === PART 5: COMPARE PER-TRANSITION GINI BETWEEN GROUPS ===
h_df = gini_human[['transition', 'gini']].rename(columns={'gini': 'human_gini'})
l_df = gini_llm[['transition', 'gini']].rename(columns={'gini': 'llm_gini'})

merged_gini = pd.merge(h_df, l_df, on='transition', how='outer').fillna(0)
merged_gini['gini_diff'] = merged_gini['human_gini'] - merged_gini['llm_gini']

# Classification logic
def classify_row(row):
    h, l = row['human_gini'], row['llm_gini']
    diff = abs(h - l)
    if h <= 0.33 and l <= 0.33:
        return 'Shared'
    elif diff > 0.3:
        if h > l:
            return 'Human-specific'
        else:
            return 'LLM-specific'
    elif h > 0.66 and l > 0.66:
        return 'Sporadic'
    else:
        return 'Mixed'

merged_gini['pattern'] = merged_gini.apply(classify_row, axis=1)
merged_gini.to_csv(OUTPUT_DIR / "transitions_gini_comparison.csv", index=False)

# === PLOTS ===

sns.set(style="whitegrid")

# Histogram of gini scores
plt.figure(figsize=(10, 5))
sns.histplot(data=merged, x='gini', hue='group', bins=20, kde=False, palette='pastel')
plt.title("Distribution of Transition Gini Scores (Human vs LLM)")
plt.xlabel("Gini Index")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hist_transition_gini.png", dpi=300)
plt.close()

# Barplot: Top transitions in each bin
plt.figure(figsize=(12, 6))
top_bar = top_by_bin.copy()
top_bar['label'] = top_bar['transition'] + '\n(' + top_bar['group'] + ')'
sns.barplot(data=top_bar, x='label', y='count', hue='gini_bin', dodge=False)
plt.title("Top Transitions by Gini Bin")
plt.xlabel("Transition")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.legend(title="Gini Bin")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "barplot_top_transitions_by_gini_bin.png", dpi=300)
plt.close()

# Boxplot of Gini Index per Participant
plt.figure(figsize=(8, 5))
sns.boxplot(data=ppt_df, x='group', y='gini', palette='pastel')
sns.stripplot(data=ppt_df, x='group', y='gini', color='black', alpha=0.6, jitter=True)
plt.title("Gini Index of Transition Usage per Participant")
plt.ylabel("Gini Index")
plt.xlabel("")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "boxplot_transition_gini_by_participant.png", dpi=300)
plt.close()

# Scatterplot: Human vs LLM transition gini
plt.figure(figsize=(7, 6))
sns.scatterplot(data=merged_gini, x='human_gini', y='llm_gini', hue='pattern', palette='Set2', s=70)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
plt.xlabel("Human Gini")
plt.ylabel("LLM Gini")
plt.title("Transition Gini Comparison by Group")
plt.legend(title="Pattern", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "scatter_transition_gini_comparison.png", dpi=300)
plt.close()
