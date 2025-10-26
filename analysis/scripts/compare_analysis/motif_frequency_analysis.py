
# motif_frequency_analysis_revised.py
# Analyse and compare 3-gram and 5-gram motifs in Human vs LLM reasoning traces

import pandas as pd
from pathlib import Path
from itertools import islice
from collections import Counter
from scipy.stats import norm

# === PATHS ===
DATA_HUMAN = Path("../analysis_data/full_human_data.csv")
DATA_LLM = Path("../analysis_data/full_llm_data_subset42.csv")
OUTPUT_DIR = Path("../analysis_results/compare/motif_analysis_new")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD DATA ===
def load_data(path, group_label):
    df = pd.read_csv(path)
    df = df[['task_id', 'ppt_id', 'high_level_code']]
    df['group'] = group_label
    return df

human = load_data(DATA_HUMAN, 'Human')
llm = load_data(DATA_LLM, 'LLM')
data = pd.concat([human, llm], ignore_index=True)

# === EXTRACT N-GRAMS PER PPT × TASK BLOCK ===
def extract_ngrams(lst, n):
    return list(zip(*(islice(lst, i, None) for i in range(n))))

motif_data = []
for (group, ppt, task), group_df in data.groupby(['group', 'ppt_id', 'task_id']):
    codes = group_df['high_level_code'].tolist()
    for n in [3, 5]:
        motifs = extract_ngrams(codes, n)
        for motif in motifs:
            motif_str = " → ".join(motif)
            motif_data.append((group, ppt, task, n, motif_str))

motif_df = pd.DataFrame(motif_data, columns=['group', 'ppt_id', 'task_id', 'ngram', 'motif'])

# === COUNT MOTIF USAGE ===
motif_counts = motif_df.groupby(['group', 'motif', 'ngram']).size().reset_index(name='count')

# === PIVOT AND CALCULATE TOTALS & PROPORTIONS ===
matrix = motif_counts.pivot_table(index=['motif', 'ngram'], columns='group', values='count', fill_value=0)
matrix.columns = [f"count_{col}" for col in matrix.columns]
matrix = matrix.reset_index()
matrix['total_count'] = matrix['count_Human'] + matrix['count_LLM']

total_by_group = motif_counts.groupby('group')['count'].sum().to_dict()
matrix['prop_Human'] = matrix['count_Human'] / total_by_group.get('Human', 1)
matrix['prop_LLM'] = matrix['count_LLM'] / total_by_group.get('LLM', 1)
matrix['prop_diff'] = matrix['prop_Human'] - matrix['prop_LLM']
matrix['group_label'] = matrix['prop_diff'].apply(lambda x: 'Human > LLM' if x > 0 else 'LLM > Human')

# === Z-TEST FOR PROPORTION DIFFERENCE ===
def compute_ztest(row):
    ph = row['prop_Human']
    pl = row['prop_LLM']
    nh = total_by_group.get('Human', 1)
    nl = total_by_group.get('LLM', 1)
    p_pool = (ph * nh + pl * nl) / (nh + nl)
    se = (p_pool * (1 - p_pool) * (1/nh + 1/nl)) ** 0.5
    if se == 0:
        return 1.0
    z = (ph - pl) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return p

matrix['p_value'] = matrix.apply(compute_ztest, axis=1)
matrix['significant'] = matrix['p_value'] < 0.05

# === SPLIT BY NGRAM ===
for n in [3, 5]:
    df_ngram = matrix[matrix['ngram'] == n].copy()
    df_ngram.to_csv(OUTPUT_DIR / f"motif_summary_{n}gram.csv", index=False)

    # === SALIENT FILTER ===
    salient = df_ngram[(df_ngram['total_count'] >= 30)].copy()
    salient.to_csv(OUTPUT_DIR / f"salient_motifs_{n}gram.csv", index=False)

    # === TOP 10 PER GROUP BASED ON USAGE PROPORTION ===
    top10_human = salient.sort_values("prop_Human", ascending=False).head(10)
    top10_llm = salient.sort_values("prop_LLM", ascending=False).head(10)

    top10_human.to_csv(OUTPUT_DIR / f"top10_human_{n}gram.csv", index=False)
    top10_llm.to_csv(OUTPUT_DIR / f"top10_llm_{n}gram.csv", index=False)
