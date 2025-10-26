# LLM Thought Transition Analysis Script
# Full pipeline: transition matrices, salience, group diff, gini, flow diagrams

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from itertools import product
from scipy.stats import norm, chi2_contingency
from collections import Counter
import warnings

# --- Paths ---
DATA_PATH = Path("../analysis_data/full_llm_data.csv")
OUTPUT_DIR = Path("../analysis_results/llm/transition_matrix_diagram")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SALIENCE_THRESHOLD = 0.10

# --- Load ---
df = pd.read_csv(DATA_PATH)
df = df.sort_values(by=["ppt_id", "task_id", "thought_id"])

# --- Build transitions ---
def extract_transitions(sub):
    return list(zip(sub["high_level_code"], sub["high_level_code"].shift(-1)))[:-1]

# ✅ Correct transition pairing within each ppt_id and task_id block
df["next_code"] = df.groupby(["ppt_id", "task_id"])["high_level_code"].shift(-1)
df_trans = df.dropna(subset=["next_code"]).copy()
df_trans["transition"] = list(zip(df_trans["high_level_code"], df_trans["next_code"]))

# --- Transition matrix (counts & probabilities) ---
def build_matrix(transitions):
    states = sorted(set([s for t in transitions for s in t]))
    mat = pd.DataFrame(0, index=states, columns=states)
    for a, b in transitions:
        mat.loc[a, b] += 1
    prob = mat.div(mat.sum(axis=1), axis=0).fillna(0)
    return mat, prob

# --- Flow diagram function (Script 1 style, no abbreviation) ---
def plot_flow(prob_matrix, title, filename, threshold=0.10):
    G = nx.DiGraph()
    edge_labels = {}
    edge_colors = []
    widths = []
    edge_list = []

    for s1, s2 in product(prob_matrix.index, repeat=2):
        w = prob_matrix.loc[s1, s2]
        if w > 0.01:
            G.add_edge(s1, s2, weight=w)
            edge_labels[(s1, s2)] = f"{w:.2f}"
            edge_list.append((s1, s2))
            edge_colors.append("black" if w >= threshold else "gray")
            widths.append(8.5 * w)

    pos = nx.circular_layout(G)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color="white", edgecolors="black", node_size=1800)
    nx.draw_networkx_labels(G, pos, font_size=12)

    nx.draw_networkx_edges(
        G, pos,
        edgelist=edge_list,                 # ✅ consistent order
        edge_color=edge_colors,
        width=widths,
        edge_cmap=None,
        style='solid',
        arrows=True,
        arrowstyle='-|>',
        arrowsize=28,
        connectionstyle='arc3,rad=0',
        alpha=0.9
    )

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# --- By group ---
conditions = {
    "overall": df_trans,
    "correct": df_trans[df_trans["accuracy"] == 1],
    "incorrect": df_trans[df_trans["accuracy"] == 0]
}
for task in sorted(df_trans["task_id"].unique()):
    conditions[f"task{task}"] = df_trans[df_trans["task_id"] == task]

# --- Store grouped matrices together ---
def append_matrix_to_csv(file, label, matrix):
    with open(file, "a") as f:
        f.write(f"# === {label} ===\n")
    matrix.to_csv(file, mode="a")
    with open(file, "a") as f:
        f.write("\n\n")

count_main_path = OUTPUT_DIR / "transition_matrix_count_main.csv"
prob_main_path = OUTPUT_DIR / "transition_matrix_prob_main.csv"
count_task_path = OUTPUT_DIR / "transition_matrix_count_by_task.csv"
prob_task_path = OUTPUT_DIR / "transition_matrix_prob_by_task.csv"

# Clear files first
for path in [count_main_path, prob_main_path, count_task_path, prob_task_path]:
    path.write_text("")

# --- Compute and save matrices + diagrams ---
for label, subdf in conditions.items():
    transitions = list(subdf["transition"])
    mat, prob = build_matrix(transitions)

    # Append matrices to grouped CSVs
    if label in ["overall", "correct", "incorrect"]:
        append_matrix_to_csv(count_main_path, label, mat)
        append_matrix_to_csv(prob_main_path, label, prob)
    elif label.startswith("task"):
        append_matrix_to_csv(count_task_path, label, mat)
        append_matrix_to_csv(prob_task_path, label, prob)

    # --- Flow diagram (Script 1 format) ---
    plot_flow(prob, f"LLM Flow Diagram - {label.capitalize()}", OUTPUT_DIR / f"flow_diagram_{label}.png", threshold=SALIENCE_THRESHOLD)

    print(f"✅ Saved matrix and flow diagram for: {label}")

