# Full script: Thought sequence analysis with stats, flow diagrams, and significant transition plots

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from itertools import product
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
import statsmodels.api as sm
import warnings

# --- Paths ---
DATA_PATH = Path("../analysis_data/full_human_data.csv")
RESULTS_DIR = Path("../analysis_results/human")
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# --- Load data ---
df = pd.read_csv(DATA_PATH, encoding="utf-8", encoding_errors="replace")
df = df[df["high_level_code"].notna()].copy()
df["next_code"] = df.groupby(["ppt_id", "task_id"])["high_level_code"].shift(-1)
df = df[df["next_code"].notna()]

all_codes = sorted(df["high_level_code"].unique())
pairs = list(product(all_codes, repeat=2))

# --- Transition count and probability matrices ---
def transition_matrix(sub):
    counts = pd.DataFrame(0, index=all_codes, columns=all_codes)
    for _, row in sub.iterrows():
        counts.loc[row["high_level_code"], row["next_code"]] += 1
    return counts

def transition_prob_matrix(counts):
    return counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

# --- Overall, correct, incorrect ---
matrices = {}
for label, subdf in {
    "overall": df,
    "correct": df[df["accuracy"] == 1],
    "incorrect": df[df["accuracy"] == 0],
}.items():
    counts = transition_matrix(subdf)
    probs = transition_prob_matrix(counts)
    matrices[label] = {"count": counts, "prob": probs}

with pd.ExcelWriter(TABLE_DIR / "human_trans_counts_main.xlsx") as writer:
    for label in matrices:
        matrices[label]["count"].to_excel(writer, sheet_name=label)
with pd.ExcelWriter(TABLE_DIR / "human_trans_probs_main.xlsx") as writer:
    for label in matrices:
        matrices[label]["prob"].to_excel(writer, sheet_name=label)

# --- By Task ---
with pd.ExcelWriter(TABLE_DIR / "human_trans_counts_by_task.xlsx") as w1, \
     pd.ExcelWriter(TABLE_DIR / "human_trans_probs_by_task.xlsx") as w2:
    for t in sorted(df["task_id"].unique()):
        sub = df[df["task_id"] == t]
        c = transition_matrix(sub)
        p = transition_prob_matrix(c)
        c.to_excel(w1, sheet_name=f"task{t}")
        p.to_excel(w2, sheet_name=f"task{t}")

# --- Z test for significant transitions ---
results = []
for (s1, s2) in pairs:
    n1 = matrices["correct"]["count"].loc[s1, s2]
    n2 = matrices["incorrect"]["count"].loc[s1, s2]
    N1 = matrices["correct"]["count"].loc[s1].sum()
    N2 = matrices["incorrect"]["count"].loc[s1].sum()
    if N1 > 0 and N2 > 0:
        p1 = n1 / N1
        p2 = n2 / N2
        p_pool = (n1 + n2) / (N1 + N2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/N1 + 1/N2))
        z = (p1 - p2) / se if se > 0 else 0
        p_val = 2 * (1 - norm.cdf(abs(z)))
        results.append({
            "from": s1, "to": s2, "correct%": round(100*p1,2), "incorrect%": round(100*p2,2),
            "z": round(z, 2), "p_value": round(p_val, 4), "significant": p_val < 0.05,
            "direction": "correct > incorrect" if p1 > p2 else "incorrect > correct"
        })

sig_df = pd.DataFrame(results)
sig_df.to_csv(TABLE_DIR / "human_transition_significance_test.csv", index=False)

# --- Flow Diagrams ---
def plot_flow(prob_matrix, title, filename, threshold=0.05):
    G = nx.DiGraph()
    abbrev = {
        'Orientation':'OR', 'Planning':'PL', 'changePlan':'CP', 'ProcessingScope':'PS',
        'MentalRepresentation':'MR', 'Hypothesis':'HYP', 'Evaluation/Monitoring':'EV',
        'DecisionMaking':'DM', 'Reflection':'REF', 'FinalRule':'FR', 'Memory':'MEM'
    }
    edge_labels = {}
    for s1, s2 in product(all_codes, repeat=2):
        w = prob_matrix.loc[s1, s2]
        if w > 0.02:
            G.add_edge(abbrev.get(s1,s1), abbrev.get(s2,s2), weight=w)
            if w >= threshold:
                edge_labels[(abbrev.get(s1,s1), abbrev.get(s2,s2))] = f"{w:.2f}"

    pos = nx.circular_layout(G)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color="white", edgecolors="black", node_size=1000)
    nx.draw_networkx_labels(G, pos, font_size=10)
    edges = G.edges(data=True)
    widths = [7*d['weight'] for (_,_,d) in edges]
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=15, edge_color='gray', width=widths)
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename)
    plt.close()

plot_flow(matrices["overall"]["prob"], "Flow: Overall", "flow_overall.png")
plot_flow(matrices["correct"]["prob"], "Flow: Correct", "flow_correct.png")
plot_flow(matrices["incorrect"]["prob"], "Flow: Incorrect", "flow_incorrect.png")

# --- Barplot for significant transitions ---
sig_pairs = sig_df[sig_df["significant"] == True][["from", "to"]].values.tolist()
sig_labels = [f"{a}→{b}" for a, b in sig_pairs]

plot_data = []
for label, subdf in {"Correct": df[df["accuracy"]==1], "Incorrect": df[df["accuracy"]==0]}.items():
    counts = transition_matrix(subdf)
    for (s1, s2) in sig_pairs:
        total = counts.loc[s1].sum()
        if total > 0:
            plot_data.append({"Transition": f"{s1}→{s2}", "Group": label, "Percentage": 100*counts.loc[s1,s2]/total})

plot_df = pd.DataFrame(plot_data)
plt.figure(figsize=(10,6))
sns.barplot(data=plot_df, x="Transition", y="Percentage", hue="Group")
plt.ylabel("Percentage")
plt.title("Relative frequency of significant transitions by accuracy group")
plt.legend(title="")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "sig_transition_barplot.png")
plt.close()

# --- Lineplot across tasks ---
task_data = []
for t in sorted(df["task_id"].unique()):
    sub = df[df["task_id"] == t]
    counts = transition_matrix(sub)
    for (s1, s2) in sig_pairs:
        total = counts.loc[s1].sum()
        if total > 0:
            task_data.append({"Task": str(t), "Transition": f"{s1}→{s2}", "Percentage": 100*counts.loc[s1,s2]/total})

task_df = pd.DataFrame(task_data)
plt.figure(figsize=(10,6))
sns.lineplot(data=task_df, x="Task", y="Percentage", hue="Transition")
plt.ylabel("Percentage")
plt.title("Trend of significant transitions across tasks")
plt.legend(title="", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(FIGURE_DIR / "sig_transition_trend.png")
plt.close()

print("✅ All sequence analyses and visuals completed.")
