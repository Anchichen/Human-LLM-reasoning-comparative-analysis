# compare_flow_diagrams.py
# Generate flow diagrams of cognitive state transitions (overall + by task)
# Author: [Your Name]

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# === CONFIG ===
INPUT_DIR = Path("../analysis_results/compare/transition_matrices")
OUTPUT_DIR = Path("../analysis_results/compare/flow_diagrams_update")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = ["human", "llm"]
TASKS = ["overall", "task1", "task2", "task3", "task4", "task5"]

# === NODE LABEL MAPPING ===
LABEL_MAP = {
    "Evaluation/Monitoring": "Evaluation",
    "ProcessingScope": "Processing"
}

# === FLOW DIAGRAM SETTINGS ===
def draw_flow_diagram(prob_matrix, title, save_path):
    # --- Label fixes ---
    label_map = {
        "Evaluation/Monitoring": "Evaluation",
        "ProcessingScope": "Processing",
    }
    prob_matrix = prob_matrix.rename(index=label_map, columns=label_map)

    # --- Build directed graph with Rule 1 (prob >= .01) ---
    G = nx.DiGraph()
    for s in prob_matrix.index:
        for t in prob_matrix.columns:
            p = float(prob_matrix.loc[s, t])
            if p >= 0.01:  # Rule 1
                G.add_edge(s, t, weight=p)

    # --- Layout & figure ---
    pos = nx.circular_layout(G)  # clean, consistent positions
    plt.figure(figsize=(9, 9))
    ax = plt.gca()
    ax.set_title(title, fontsize=14, fontname="Arial")
    ax.set_facecolor("white")

    # Nodes
    nx.draw_networkx_nodes(
        G, pos, node_color="white", edgecolors="black",
        node_size=1800, linewidths=1.2
    )
    nx.draw_networkx_labels(
        G, pos, font_size=11, font_family="Arial", font_color="black"
    )

    # Split edges by salience (Rule 3)
    edges = list(G.edges(data=True))
    black_edges = [(u, v, d) for u, v, d in edges if d["weight"] >= 0.10]
    grey_edges  = [(u, v, d) for u, v, d in edges if 0.01 <= d["weight"] < 0.10]

    # Helper to draw straight edges with a given colour
    def draw_edges(edgelist, color, width_scale, arrow_size):
        if not edgelist:
            return
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v) for u, v, _ in edgelist],
            width=[max(1.0, d["weight"] * width_scale) for _, _, d in edgelist],
            edge_color=color,
            arrows=True, arrowstyle="-|>", arrowsize=arrow_size,
            connectionstyle="arc3,rad=0.0",  # straight for all edges
            alpha=0.95
        )

    # Draw edges (Rule 3 colours)
    draw_edges(black_edges, "black",     width_scale=12, arrow_size=34)
    draw_edges(grey_edges,  "#D9D9D9",   width_scale=10, arrow_size=28)

    # Edge labels for prob >= .05 (Rule 2)
    edge_labels = {
        (u, v): f"{d['weight']:.2f}"
        for u, v, d in edges if d["weight"] >= 0.05
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels,
        font_size=9, font_family="Arial",
        bbox=dict(alpha=0, color="none")
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
