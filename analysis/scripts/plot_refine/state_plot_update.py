import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

# --- Calculate proportions ---
state_counts = df_combined.groupby(["group", "high_level_code"]).size().reset_index(name="count")
state_total = df_combined.groupby("group").size().reset_index(name="total")
state_prop = pd.merge(state_counts, state_total, on="group")
state_prop["proportion"] = state_prop["count"] / state_prop["total"]

# Ensure all states exist in both groups
all_states = df_combined["high_level_code"].unique()
full_index = pd.MultiIndex.from_product([["Human", "LLM"], all_states], names=["group", "high_level_code"])
state_prop = state_prop.set_index(["group", "high_level_code"]).reindex(full_index, fill_value=0).reset_index()

# --- Custom labels (with two-line splits where needed) ---
pretty_labels = {
    "MentalRepresentation": "Mental\nRepresentation",
    "DecisionMaking": "Decision\nMaking",
    "Evaluation_Monitoring": "Evaluation",   # underscore form
    "Evaluation/Monitoring": "Evaluation",   # slash form (your data)
    "ProcessingScope": "Processing"
}
state_prop["high_level_code"] = state_prop["high_level_code"].replace(pretty_labels)


# --- Order: LLM ascending, Human-only first ---
llm_order = (
    state_prop[state_prop["group"] == "LLM"]
    .sort_values("proportion")["high_level_code"]
    .tolist()
)
human_only = [
    s for s in state_prop["high_level_code"].unique()
    if s not in df_llm["high_level_code"].replace(pretty_labels).unique()
]
human_ordered = (
    state_prop[(state_prop["group"] == "Human") & (state_prop["high_level_code"].isin(human_only))]
    .sort_values("proportion", ascending=False)["high_level_code"]
    .tolist()
)
llm_order = human_ordered + [s for s in llm_order if s not in human_only]

# --- Plot ---
plt.rcParams["font.family"] = "Arial"
plt.figure(figsize=(10, 5))
sns.barplot(
    data=state_prop,
    x="high_level_code",
    y="proportion",
    hue="group",
    order=llm_order,
    palette=["#4E79A7", "#F28E2B"]
)
plt.title("Proportion of High-Level Cognitive States", fontsize=15)
plt.xlabel("")
plt.ylabel("Proportion", fontsize=12)
plt.xticks(rotation=0, fontsize=10, ha="center")
plt.yticks(fontsize=10)
plt.legend(title="Group", fontsize=10, title_fontsize=10)
plt.tight_layout()

# Save
plt.savefig(RESULT_DIR / "state_proportions_barplot.png", dpi=300)
plt.close()

print("✅ State proportions barplot saved!")

