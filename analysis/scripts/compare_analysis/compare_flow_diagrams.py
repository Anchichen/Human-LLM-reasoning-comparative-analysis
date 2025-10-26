# compare_flow_diagrams.py
# Generate flow diagrams of cognitive state transitions (overall + by task)
# Author: [Your Name]

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# === CONFIG ===
INPUT_DIR = Path("../analysis_results/compare/transition_matrices")
OUTPUT_DIR = Path("../analysis_results/compare/flow_diagrams")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = ["human", "llm"]
TASKS = ["overall", "task1", "task2", "task3", "task4", "task5"]

# === FLOW DIAGRAM SETTINGS ===
def draw_flow_diagram(prob_matrix, title, save_path):
    G = nx.DiGraph()

    # Add edges with probabilities >= 0.01
    for source in prob_matrix.index:
        for target in prob_matrix.columns:
            prob = prob_matrix.loc[source, target]
            if prob >= 0.01:
                G.add_edge(source, target, weight=prob)

    pos = nx.circular_layout(G)

    plt.figure(figsize=(9, 9))
    ax = plt.gca()
    ax.set_title(title, fontsize=14)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color="white",
        edgecolors="black",
        node_size=1800
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=11,
        font_family="sans-serif"
    )

    # Separate edges into black (salient) and grey
    edges = G.edges(data=True)
    black_edges = [(u, v) for u, v, d in edges if d["weight"] >= 0.10]
    grey_edges = [(u, v) for u, v, d in edges if d["weight"] < 0.10]

    # Draw black salient edges
    nx.draw_networkx_edges(
        G, pos,
        edgelist=black_edges,
        width=[G[u][v]["weight"] * 10 for u, v in black_edges],
        edge_color="black",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=27,
        connectionstyle="arc3,rad=0",
        alpha=0.9,
        style="solid"
    )

    # Draw grey lighter edges
    nx.draw_networkx_edges(
        G, pos,
        edgelist=grey_edges,
        width=[G[u][v]["weight"] * 10 for u, v in grey_edges],
        edge_color="lightgrey",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=27,
        connectionstyle="arc3,rad=0",
        alpha=0.9,
        style="solid"
    )

    # Draw edge labels for edges with prob >= 0.05
    edge_labels = {
        (u, v): f"{d['weight']:.2f}"
        for u, v, d in edges if d["weight"] >= 0.05
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# === MAIN LOOP TO LOAD MATRICES AND PLOT ===
for group in GROUPS:
    for task in TASKS:
        filename = f"{group}_{task}_transition_prob.csv"
        filepath = INPUT_DIR / filename
        if not filepath.exists():
            print(f"⚠️ File not found: {filename}")
            continue

        df = pd.read_csv(filepath, index_col=0)
        title = f"{'LLM' if group == 'llm' else 'Human'} Flow Diagram – {task.capitalize()}"
        save_file = OUTPUT_DIR / f"{group}_{task}_flow_diagram.png"

        draw_flow_diagram(df, title, save_file)

print("✅ Flow diagrams saved to:", OUTPUT_DIR)


