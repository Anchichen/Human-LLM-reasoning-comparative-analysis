# clustering_analysis.py
# Identify reasoning strategy clusters in Human + LLM using high-level state and transition profiles

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

# === PATHS ===
HUMAN_PATH = Path("../analysis_data/full_human_data.csv")
LLM_PATH = Path("../analysis_data/full_llm_data_subset42.csv")
OUTPUT_DIR = Path("../analysis_results/compare/clustering/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD DATA ===
def load_data(path, group):
    df = pd.read_csv(path)
    df['group'] = group
    return df[['ppt_id', 'task_id', 'high_level_code', 'group']]

df_h = load_data(HUMAN_PATH, 'Human')
df_l = load_data(LLM_PATH, 'LLM')
df = pd.concat([df_h, df_l], ignore_index=True)

# === STATE FREQUENCY FEATURES ===
state_df = df.groupby(['ppt_id', 'high_level_code', 'group']).size().reset_index(name='count')
state_wide = state_df.pivot_table(index=['ppt_id', 'group'], columns='high_level_code', values='count', fill_value=0).reset_index()

# === TRANSITION FEATURES ===
all_states = sorted(df['high_level_code'].unique())
transitions = []
for (ppt, task), block in df.groupby(['ppt_id', 'task_id']):
    codes = block['high_level_code'].tolist()
    group = block['group'].iloc[0]
    for a, b in zip(codes, codes[1:]):
        transitions.append((ppt, group, a, b))
trans_df = pd.DataFrame(transitions, columns=['ppt_id', 'group', 'source', 'target'])
trans_df['pair'] = trans_df['source'] + " → " + trans_df['target']
trans_counts = trans_df.groupby(['ppt_id', 'pair', 'group']).size().reset_index(name='count')
trans_wide = trans_counts.pivot_table(index=['ppt_id', 'group'], columns='pair', values='count', fill_value=0).reset_index()

# === SHARED PIVOT IDS ===
shared_ids = set(state_wide['ppt_id']) & set(trans_wide['ppt_id'])
state_wide = state_wide[state_wide['ppt_id'].isin(shared_ids)]
trans_wide = trans_wide[trans_wide['ppt_id'].isin(shared_ids)]

# === COMBINED FEATURES ===
combined = pd.merge(state_wide, trans_wide, on=['ppt_id', 'group'], suffixes=('_state', '_trans'))
features_state = state_wide.drop(columns=['ppt_id', 'group'])
features_trans = trans_wide.drop(columns=['ppt_id', 'group'])
features_combined = combined.drop(columns=['ppt_id', 'group'])

# === STANDARDIZE ===
scaler = StandardScaler()
X_state = scaler.fit_transform(features_state)
X_trans = scaler.fit_transform(features_trans)
X_combined = scaler.fit_transform(features_combined)

# === CLUSTERING FUNCTION ===
def run_clustering(X, meta_df, name, feature_df):
    sil_scores = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        sil_scores.append((k, sil))
    best_k = max(sil_scores, key=lambda x: x[1])[0]
    print(f"✅ {name}: Best k = {best_k}")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    meta_df['cluster'] = kmeans.fit_predict(X)

    # PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(X)
    meta_df['PC1'] = proj[:, 0]
    meta_df['PC2'] = proj[:, 1]

    # Proportions
    proportions = pd.DataFrame(X, columns=feature_df.columns)
    meta_df = pd.concat([meta_df.reset_index(drop=True), proportions.reset_index(drop=True)], axis=1)

    # Sort meta_df
    meta_df['ppt_num'] = meta_df['ppt_id'].str.extract(r'(\d+)').astype(int)
    meta_df = meta_df.sort_values(by=['group', 'ppt_num']).drop(columns='ppt_num')

    # Export membership
    meta_df = meta_df[['ppt_id', 'group', 'cluster', 'PC1', 'PC2'] + list(feature_df.columns)]
    meta_df.to_csv(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_cluster_membership.csv", index=False)

    # Composition
    comp = meta_df.groupby(['cluster', 'group']).size().unstack(fill_value=0)
    comp['Total'] = comp.sum(axis=1)
    comp.to_csv(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_group_composition.csv")

    # Cluster profile
    feature_cols = feature_df.columns
    profile = meta_df.groupby('cluster')[feature_cols].mean()
    profile = profile.round(2)

    # Top 5 features
    top5_df = []
    for c in profile.index:
        top_feats = profile.loc[c].sort_values(ascending=False).head(5)
        row = {'cluster': c}
        for i, feat in enumerate(top_feats.index):
            row[f"Top {i+1}"] = feat
        top5_df.append(row)
    top5_df = pd.DataFrame(top5_df)

    # Combine into one CSV with blank row separation
    with open(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_cluster_profiles.csv", "w") as f:
        profile.reset_index().to_csv(f, index=False)
        f.write("\n\n")
        top5_df.to_csv(f, index=False)

    # PCA Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=meta_df, x='PC1', y='PC2', hue='cluster', style='group', s=80, palette='tab10')
    plt.title(f"{name} Reasoning Clusters (PCA Projection)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"pca_clusters_{name.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()

# === RUN THREE CLUSTERINGS ===
meta_state = state_wide[['ppt_id', 'group']].copy()
meta_trans = trans_wide[['ppt_id', 'group']].copy()
meta_comb = combined[['ppt_id', 'group']].copy()

run_clustering(X_state, meta_state, "State Frequency", features_state)
run_clustering(X_trans, meta_trans, "Transition Pattern", features_trans)
run_clustering(X_combined, meta_comb, "Combined Profile", features_combined)

print("✅ All clustering analyses complete.")


