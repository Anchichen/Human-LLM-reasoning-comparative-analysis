import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency, mannwhitneyu, ks_2samp, zscore
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

# --- Paths ---
DATA_DIR = Path("../analysis_data")
HUMAN_FILE = DATA_DIR / "full_human_data.csv"
LLM_FILE = DATA_DIR / "full_llm_data_subset42.csv"
RESULT_DIR = Path("../analysis_results/compare/section1_descriptive")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load and label data ---
df_human = pd.read_csv(HUMAN_FILE)
df_llm = pd.read_csv(LLM_FILE)
df_human["group"] = "Human"
df_llm["group"] = "LLM"
df_combined = pd.concat([df_human, df_llm], ignore_index=True)

# =============================
# 1. Cognitive State Proportions
# =============================
state_counts = df_combined.groupby(["group", "high_level_code"]).size().reset_index(name="count")
state_total = df_combined.groupby("group").size().reset_index(name="total")
state_prop = pd.merge(state_counts, state_total, on="group")
state_prop["proportion"] = state_prop["count"] / state_prop["total"]

# Ensure all states are present for both groups
all_states = df_combined["high_level_code"].unique()
full_index = pd.MultiIndex.from_product([["Human", "LLM"], all_states], names=["group", "high_level_code"])
state_prop = state_prop.set_index(["group", "high_level_code"]).reindex(full_index, fill_value=0).reset_index()

# Save sorted table
state_prop = state_prop.sort_values(by=["group", "proportion"], ascending=[True, False])
state_prop.to_csv(RESULT_DIR / "state_proportions_by_group.csv", index=False)

# === Plot with custom order ===
# Sort by LLM ascending, then Human-only on left
llm_order = (
    state_prop[state_prop["group"] == "LLM"]
    .sort_values("proportion")["high_level_code"]
    .tolist()
)
# Find human-only states and sort them by proportion (descending)
human_only = [
    s for s in all_states if s not in df_llm["high_level_code"].unique()
]
human_ordered = (
    state_prop[(state_prop["group"] == "Human") & (state_prop["high_level_code"].isin(human_only))]
    .sort_values("proportion", ascending=False)["high_level_code"]
    .tolist()
)

# Combine final order
llm_order = human_ordered + [s for s in llm_order if s not in human_only]

plt.figure(figsize=(10, 5))
sns.barplot(data=state_prop, x="high_level_code", y="proportion", hue="group", order=llm_order)
plt.title("Proportion of High-Level Cognitive States")
plt.xlabel("")  # Remove x label
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(RESULT_DIR / "state_proportions_barplot.png")
plt.close()

# =============================
# 2. Token Count Violin Plot
# =============================
plt.figure(figsize=(7, 5))
sns.violinplot(data=df_combined, x="group", y="token_count", inner="box", density_norm="count")
plt.title("Token Count per Thought")
plt.xlabel("")
plt.ylabel("Token Count")
plt.figtext(0.5, -0.05,
    "Violin shows token distribution. Width = density. Box = IQR. Line = median.",
    wrap=True, ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(RESULT_DIR / "token_count_violinplot.png", bbox_inches="tight")
plt.close()

# ============================
# 3. Step Count Violin Plot
# ============================
step_df = df_combined.groupby(["ppt_id", "group", "task_id"]).size().reset_index(name="step_count")
plt.figure(figsize=(7, 5))
sns.violinplot(data=step_df, x="group", y="step_count", inner="box", density_norm="count")
plt.title("Step Count per Task")
plt.xlabel("")
plt.ylabel("Step Count")
plt.figtext(0.5, -0.05,
    "Violin shows step count per task. Width = density. Box = IQR. Line = median.",
    wrap=True, ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(RESULT_DIR / "step_count_violinplot.png", bbox_inches="tight")
plt.close()

# ============================
# 4. Gini Index (Sorted Format + Overall)
# ============================
def gini(x):
    if len(x) == 0 or np.sum(x) == 0:
        return 0
    sorted_x = np.sort(x)
    n = len(x)
    cum_x = np.cumsum(sorted_x)
    return (n + 1 - 2 * np.sum(cum_x) / cum_x[-1]) / n

def compute_gini(df):
    freq = (
        df.groupby("ppt_id")["high_level_code"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    return freq.apply(gini, axis=1)

gini_human = compute_gini(df_human).reset_index()
gini_human["group"] = "Human"
gini_llm = compute_gini(df_llm).reset_index()
gini_llm["group"] = "LLM"

gini_all = pd.concat([gini_human, gini_llm])
gini_all.columns = ["ppt_id", "gini", "group"]

# Sort: Human (p1–p42), then LLM (llm1–llm42)
def sort_ppt(ppt_id):
    if ppt_id.startswith("p"):
        return (0, int(ppt_id[1:]))
    elif ppt_id.startswith("llm"):
        return (1, int(ppt_id[3:]))
    return (2, 0)

gini_all["sort_key"] = gini_all["ppt_id"].apply(sort_ppt)
gini_all = gini_all.sort_values(["group", "sort_key"]).drop(columns="sort_key")

# Add overall Gini from group-level proportions
overall_gini_h = gini(df_human["high_level_code"].value_counts(normalize=True).values)
overall_gini_l = gini(df_llm["high_level_code"].value_counts(normalize=True).values)
gini_all = pd.concat([
    gini_all,
    pd.DataFrame([
        {"ppt_id": "OVERALL", "gini": overall_gini_h, "group": "Human"},
        {"ppt_id": "OVERALL", "gini": overall_gini_l, "group": "LLM"},
    ])
])

gini_all.to_csv(RESULT_DIR / "gini_scores_by_group.csv", index=False)


# Boxplot (Gini per group)
gini_long = gini_all[gini_all["ppt_id"] != "OVERALL"]
plt.figure(figsize=(7, 5))
sns.boxplot(data=gini_long, x="group", y="gini")
plt.title("Gini Index of State Usage per Participant")
plt.tight_layout()
plt.savefig(RESULT_DIR / "gini_index_boxplot.png")
plt.close()


# ============================
# 5. Summary Stats + Tests
# ============================
summary = {}

# Token and Step stats
for grp in ["Human", "LLM"]:
    d_tok = df_combined[df_combined["group"] == grp]["token_count"]
    d_step = step_df[step_df["group"] == grp]["step_count"]
    summary[f"{grp}_token_mean"] = round(d_tok.mean(), 2)
    summary[f"{grp}_token_std"] = round(d_tok.std(), 2)
    summary[f"{grp}_step_mean"] = round(d_step.mean(), 2)
    summary[f"{grp}_step_std"] = round(d_step.std(), 2)

# Gini stats (from gini_all)
gini_stats = gini_all[gini_all["ppt_id"] != "OVERALL"]
summary["Human_gini_mean"] = round(gini_stats[gini_stats["group"] == "Human"]["gini"].mean(), 4)
summary["Human_gini_std"] = round(gini_stats[gini_stats["group"] == "Human"]["gini"].std(), 4)
summary["LLM_gini_mean"] = round(gini_stats[gini_stats["group"] == "LLM"]["gini"].mean(), 4)
summary["LLM_gini_std"] = round(gini_stats[gini_stats["group"] == "LLM"]["gini"].std(), 4)

# Statistical tests
summary["token_MW_p"] = float(mannwhitneyu(df_human["token_count"], df_llm["token_count"]).pvalue)
summary["step_MW_p"] = float(mannwhitneyu(
    step_df[step_df["group"] == "Human"]["step_count"],
    step_df[step_df["group"] == "LLM"]["step_count"]
).pvalue)

summary["token_KS_p"] = float(ks_2samp(df_human["token_count"], df_llm["token_count"]).pvalue)
summary["step_KS_p"] = float(ks_2samp(
    step_df[step_df["group"] == "Human"]["step_count"],
    step_df[step_df["group"] == "LLM"]["step_count"]
).pvalue)

# Chi-squared
state_ct = pd.crosstab(df_combined["high_level_code"], df_combined["group"])
chi2, p_chi, dof, expected = chi2_contingency(state_ct)
summary["state_chi2_stat"] = round(chi2, 4)
summary["state_chi2_p"] = float(p_chi)
summary["state_chi2_dof"] = dof

# Cosine/Jaccard
freq_h = df_human["high_level_code"].value_counts(normalize=True).sort_index()
freq_l = df_llm["high_level_code"].value_counts(normalize=True).sort_index()
union_idx = freq_h.index.union(freq_l.index)
v_h = freq_h.reindex(union_idx, fill_value=0)
v_l = freq_l.reindex(union_idx, fill_value=0)
summary["cosine_similarity"] = round(cosine_similarity([v_h], [v_l])[0][0], 4)
summary["jaccard_similarity"] = round(jaccard_score((v_h > 0).astype(int), (v_l > 0).astype(int)), 4)

# Save
pd.Series(summary).to_csv(RESULT_DIR / "summary_statistics.csv")


print("✅ All refinements complete!")
print(f"📁 Final outputs saved to: {RESULT_DIR}")

