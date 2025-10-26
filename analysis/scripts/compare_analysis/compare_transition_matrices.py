# compare_transition_matrices.py
# Author: [Your Name]
# Purpose: Build transition count and probability matrices for Human vs LLM, within run & task
# Output: CSVs for each group (human, llm), for overall and each task

import pandas as pd
import numpy as np
from pathlib import Path

# === PATH SETUP ===
DATA_HUMAN = Path("../analysis_data/full_human_data.csv")
DATA_LLM = Path("../analysis_data/full_llm_data_subset42.csv")
OUT_DIR = Path("../analysis_results/compare/transition_matrices")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD DATA ===
df_human = pd.read_csv(DATA_HUMAN)
df_llm = pd.read_csv(DATA_LLM)

# === FUNCTION TO CALCULATE TRANSITION COUNTS ===
def get_transition_matrices(df, group_label):
    count_matrices = {}
    prob_matrices = {}

    # Get full list of unique high-level codes
    all_states = sorted(df["high_level_code"].dropna().unique())

    # By task
    for task_id in sorted(df["task_id"].unique()):
        df_task = df[df["task_id"] == task_id]

        # Build list of transitions within each ppt_id and task block
        transitions = []
        for ppt_id, df_sub in df_task.groupby("ppt_id"):
            codes = df_sub.sort_values("thought_id")["high_level_code"].tolist()
            transitions += list(zip(codes, codes[1:]))

        # Create count matrix
        trans_counts = pd.DataFrame(transitions, columns=["source", "target"]).value_counts().reset_index(name="count")
        matrix_count = trans_counts.pivot_table(index="source", columns="target", values="count", fill_value=0)
        matrix_count = matrix_count.reindex(index=all_states, columns=all_states, fill_value=0)
        count_matrices[f"{group_label}_task{task_id}"] = matrix_count

        # Convert to probability matrix
        matrix_prob = matrix_count.div(matrix_count.sum(axis=1), axis=0).fillna(0)
        prob_matrices[f"{group_label}_task{task_id}"] = matrix_prob

    # Overall matrix across all tasks (but still within task blocks)
    transitions_all = []
    for (ppt_id, task_id), df_sub in df.groupby(["ppt_id", "task_id"]):
        codes = df_sub.sort_values("thought_id")["high_level_code"].tolist()
        transitions_all += list(zip(codes, codes[1:]))

    trans_counts_all = pd.DataFrame(transitions_all, columns=["source", "target"]).value_counts().reset_index(name="count")
    matrix_count_all = trans_counts_all.pivot_table(index="source", columns="target", values="count", fill_value=0)
    matrix_count_all = matrix_count_all.reindex(index=all_states, columns=all_states, fill_value=0)
    count_matrices[f"{group_label}_overall"] = matrix_count_all

    matrix_prob_all = matrix_count_all.div(matrix_count_all.sum(axis=1), axis=0).fillna(0)
    prob_matrices[f"{group_label}_overall"] = matrix_prob_all

    return count_matrices, prob_matrices

# === APPLY TO HUMAN AND LLM ===
human_counts, human_probs = get_transition_matrices(df_human, "human")
llm_counts, llm_probs = get_transition_matrices(df_llm, "llm")

# === SAVE MATRICES TO CSV ===
for name, matrix in human_counts.items():
    matrix.to_csv(OUT_DIR / f"{name}_transition_count.csv")

for name, matrix in human_probs.items():
    matrix.to_csv(OUT_DIR / f"{name}_transition_prob.csv")

for name, matrix in llm_counts.items():
    matrix.to_csv(OUT_DIR / f"{name}_transition_count.csv")

for name, matrix in llm_probs.items():
    matrix.to_csv(OUT_DIR / f"{name}_transition_prob.csv")

print("✅ Transition matrices saved successfully.")
