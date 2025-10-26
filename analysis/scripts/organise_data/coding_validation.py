import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from collections import Counter
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import networkx as nx
from itertools import combinations

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "analysis" / "analysis_data"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "analysis_results" / "coding_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_A = DATA_DIR / "full_llm_data.csv"
FILE_B = DATA_DIR / "full_llm_data_raterB.csv"

# === Load Data ===
df_a = pd.read_csv(FILE_A)
df_b = pd.read_csv(FILE_B)

# === Merge on task_id, run_id, thought_id ===
df_merge = pd.merge(
    df_a[["task_id", "run_id", "thought_id", "high_level_code"]],
    df_b[["task_id", "run_id", "thought_id", "high_level_code"]],
    on=["task_id", "run_id", "thought_id"],
    suffixes=("_a", "_b")
)
df_merge = df_merge.dropna(subset=["high_level_code_a", "high_level_code_b"])
print(f"✅ Merged {len(df_merge)} valid coded thoughts.")

# === Agreement Rate and Cohen’s Kappa ===
agreement = (df_merge["high_level_code_a"] == df_merge["high_level_code_b"]).mean()
kappa = cohen_kappa_score(df_merge["high_level_code_a"], df_merge["high_level_code_b"])
irr_stats = pd.DataFrame({
    "metric": ["agreement_rate", "cohen_kappa"],
    "value": [agreement, kappa]
})
irr_stats.to_csv(OUTPUT_DIR / "irr_stats.csv", index=False)
print("✅ Saved irr_stats.csv")

# === Fix: aligned label vector construction ===
all_labels = sorted(set(df_merge["high_level_code_a"]) | set(df_merge["high_level_code_b"]))
label_index = {label: i for i, label in enumerate(all_labels)}

# === Intersection over Union (IoU) and Modified Hausdorff Distance (MHD) ===
def label_to_vec(labels, label_index):
    vec = np.zeros(len(label_index))
    for label in labels:
        vec[label_index[label]] += 1
    return vec

vec_a = label_to_vec(df_merge["high_level_code_a"], label_index)
vec_b = label_to_vec(df_merge["high_level_code_b"], label_index)

# Normalize
vec_a = vec_a / vec_a.sum()
vec_b = vec_b / vec_b.sum()

# Compute IoU and Modified Hausdorff Distance
intersection = np.minimum(vec_a, vec_b).sum()
union = np.maximum(vec_a, vec_b).sum()
iou = intersection / union

mhd = np.linalg.norm(vec_a - vec_b)

iou_mhd_df = pd.DataFrame({
    "metric": ["IoU", "ModifiedHausdorffDistance"],
    "value": [iou, mhd]
})
iou_mhd_df.to_csv(OUTPUT_DIR / "iou_mhd_scores.csv", index=False)
print("✅ Saved iou_mhd_scores.csv")

# === Graph Edit Distance (GED) ===
def build_transition_graph(df):
    graph = nx.DiGraph()
    grouped = df.sort_values(["task_id", "run_id", "thought_id"]).groupby(["task_id", "run_id"])
    for _, group in grouped:
        codes = list(group["high_level_code"])
        for a, b in zip(codes, codes[1:]):
            graph.add_edge(a, b, weight=graph.get_edge_data(a, b, default={"weight": 0})["weight"] + 1)
    return graph

G_a = build_transition_graph(df_a)
G_b = build_transition_graph(df_b)

ged = nx.graph_edit_distance(G_a, G_b)
with open(OUTPUT_DIR / "ged_summary.txt", "w") as f:
    f.write(f"Graph Edit Distance (GED) between Rater A and Rater B transition graphs: {ged:.4f}")
print("✅ Saved ged_summary.txt")

print("\n🎉 Coding validation complete!")
