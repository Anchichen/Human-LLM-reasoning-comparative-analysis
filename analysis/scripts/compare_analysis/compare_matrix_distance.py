# matrix_distance_analysis.py
# Compare Human vs LLM transition matrices with advanced distance metrics

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy
from pathlib import Path
from itertools import product
import networkx as nx
import random
import os

# === PATHS ===
BASE_DIR = Path("../analysis_results/compare")
MATRIX_DIR = BASE_DIR / "transition_matrices"
OUTPUT_DIR = BASE_DIR / "matrix_distance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD ===
def load_matrix(path):
    return pd.read_csv(path, index_col=0)

prob_h = load_matrix(MATRIX_DIR / "human_overall_transition_prob.csv")
prob_l = load_matrix(MATRIX_DIR / "llm_overall_transition_prob.csv")

all_states = sorted(set(prob_h.index) | set(prob_l.index))
prob_h = prob_h.reindex(index=all_states, columns=all_states, fill_value=0)
prob_l = prob_l.reindex(index=all_states, columns=all_states, fill_value=0)

vec_h = prob_h.values.flatten()
vec_l = prob_l.values.flatten()

# === Frobenius ===
frob = np.linalg.norm(vec_h - vec_l)

# === KL Divergence (row-wise) ===
kl_total = 0
row_kl = {}
for s in all_states:
    p = prob_h.loc[s].replace(0, 1e-10)
    q = prob_l.loc[s].replace(0, 1e-10)
    kl = entropy(p, q)
    row_kl[s] = kl
    kl_total += kl

pd.Series(row_kl).sort_values(ascending=False).to_csv(OUTPUT_DIR / "kl_divergence_per_state.csv")

# === Heatmap ===
diff = prob_h - prob_l
plt.figure(figsize=(10, 8))
sns.heatmap(diff, cmap="RdBu_r", center=0, annot=True, fmt=".2f", cbar_kws={"label": "Human - LLM"})
plt.title("Transition Probability Difference Matrix (Human - LLM)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "transition_matrix_difference_heatmap.png", dpi=300)
plt.close()

# === Permutation test (Frobenius significance) ===
def permutation_frobenius(h_file, l_file, n_perm=500):
    all_files = list(Path("analysis_data").glob("full_*.csv"))
    df_h = pd.read_csv(h_file)
    df_l = pd.read_csv(l_file)
    df_h['group'] = 'Human'
    df_l['group'] = 'LLM'
    combined = pd.concat([df_h, df_l])

    scores = []
    for _ in range(n_perm):
        shuffled = combined.copy()
        shuffled['group'] = np.random.permutation(shuffled['group'])
        ph = shuffled[shuffled.group == 'Human']
        pl = shuffled[shuffled.group == 'LLM']

        def get_transitions(df):
            mat = pd.DataFrame(0, index=all_states, columns=all_states)
            for (pid, tid), block in df.groupby(['ppt_id', 'task_id']):
                seq = block['high_level_code'].tolist()
                for a, b in zip(seq, seq[1:]):
                    mat.loc[a, b] += 1
            return mat.div(mat.sum(axis=1).replace(0, 1), axis=0).fillna(0)

        p1 = get_transitions(ph).values.flatten()
        p2 = get_transitions(pl).values.flatten()
        scores.append(np.linalg.norm(p1 - p2))
    return scores

perm_scores = permutation_frobenius("../analysis_data/full_human_data.csv", "../analysis_data/full_llm_data_subset42.csv")
p_perm = np.mean([s >= frob for s in perm_scores])

# === Graph Edit Distance ===
def matrix_to_graph(matrix):
    G = nx.DiGraph()
    for a, b in product(matrix.index, matrix.columns):
        w = matrix.loc[a, b]
        if w > 0:
            G.add_edge(a, b, weight=w)
    return G

G_h = matrix_to_graph(prob_h)
G_l = matrix_to_graph(prob_l)
try:
    ged = nx.graph_edit_distance(G_h, G_l)
except:
    ged = np.nan

# === Output summary ===
summary = pd.DataFrame({
    'metric': ['Frobenius', 'KL Divergence', 'Permutation p-value', 'Graph Edit Distance'],
    'value': [frob, kl_total, p_perm, ged]
})
summary.to_csv(OUTPUT_DIR / "matrix_distance_results.csv", index=False)

print("✅ Matrix comparison complete. Saved to matrix_distance folder.")



# === ANALYZE TRANSITION DIFFERENCES PER SOURCE STATE ===
SOURCE_OUTPUT_DIR = OUTPUT_DIR / "source_state_comparisons"
SOURCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_states = prob_h.index.tolist()

for source in all_states:
    human_row = prob_h.loc[source]
    llm_row = prob_l.loc[source]
    diff_row = human_row - llm_row

    comparison_df = pd.DataFrame({
        "Target State": all_states,
        "Human Prob": human_row.values,
        "LLM Prob": llm_row.values,
        "Difference (Human - LLM)": diff_row.values
    })

    # Sort by Human prob for readability
    comparison_df = comparison_df.sort_values("Human Prob", ascending=False)

    # Save CSV
    # Sanitize filename
    safe_source = source.replace("/", "_").replace(" ", "_")

    comparison_df.to_csv(SOURCE_OUTPUT_DIR / f"{safe_source}_transition_comparison.csv", index=False)


    # Plot bar chart
    x = np.arange(len(all_states))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, human_row.values, width, label="Human", color="steelblue")
    ax.bar(x + width/2, llm_row.values, width, label="LLM", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(all_states, rotation=45, ha='right')
    ax.set_ylabel("Transition Probability")
    ax.set_title(f"Transition Flow from '{source}': Human vs LLM")
    ax.legend()
    plt.tight_layout()
    plt.savefig(SOURCE_OUTPUT_DIR / f"{safe_source}_transition_comparison.png", dpi=300)
    plt.close()

