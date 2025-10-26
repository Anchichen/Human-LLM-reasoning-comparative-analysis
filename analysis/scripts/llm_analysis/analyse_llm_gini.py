import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import statsmodels.formula.api as smf

from scipy.stats import skew

# --- Paths ---
DATA_PATH = Path("../analysis_data/full_llm_data.csv")
OUTPUT_DIR = Path("../analysis_results/llm/gini")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_PPT_OUTPUT = OUTPUT_DIR / "llm_gini_scores.csv"
TABLE_TASK_OUTPUT = OUTPUT_DIR / "llm_gini_by_task.csv"
TABLE_OVERALL_DOMINANT = OUTPUT_DIR / "llm_gini_dominant_states_overall.csv"
TABLE_BIN_DOMINANT = OUTPUT_DIR / "llm_gini_dominant_states_by_bin.csv"
FIGURE_GINI_BOXPLOT = OUTPUT_DIR / "llm_gini_step_vs_token_boxplot.png"
FIGURE_GINI_LINE = OUTPUT_DIR / "llm_gini_task_lineplot.png"
FIGURE_ACC_BOXPLOT = OUTPUT_DIR / "llm_gini_accuracy_boxplot.png"
REGRESSION_OUT = OUTPUT_DIR / "llm_gini_accuracy_regression.txt"

# --- Gini calculation ---
def gini_coefficient(x):
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0
    sorted_x = np.sort(x)
    n = len(x)
    cum_x = np.cumsum(sorted_x)
    return (n + 1 - 2 * np.sum(cum_x) / cum_x[-1]) / n

# --- Load dataset ---
df = pd.read_csv(DATA_PATH)

# --- Gini (step/token) and dominant state by ppt_id ---
records = []
most_used_step_overall = []
most_used_token_overall = []
for ppt_id, group in df.groupby("ppt_id"):
    step_counts = group["high_level_code"].value_counts()
    token_counts = group.groupby("high_level_code")["token_count"].sum()

    gini_step = gini_coefficient(step_counts.values)
    gini_token = gini_coefficient(token_counts.values)

    top_step = step_counts.idxmax()
    top_token = token_counts.idxmax()

    records.append({
        "ppt_id": ppt_id,
        "gini_step": gini_step,
        "gini_token": gini_token,
        "most_used_step_state": top_step,
        "most_used_token_state": top_token
    })
    most_used_step_overall.append(top_step)
    most_used_token_overall.append(top_token)

df_gini = pd.DataFrame(records).sort_values("ppt_id").reset_index(drop=True)

# --- Overall Gini and dominant state counts ---
overall_step = gini_coefficient(df["high_level_code"].value_counts().values)
overall_token = gini_coefficient(df.groupby("high_level_code")["token_count"].sum().values)

# === Save overall step-based Gini statistics ===
overall_step_counts = df["high_level_code"].value_counts()
step_gini_stats = {
    "overall_gini_step": overall_step,
    "mean_step_count": overall_step_counts.mean(),
    "std_step_count": overall_step_counts.std(),
    "skew_step_count": skew(overall_step_counts)
}
pd.DataFrame([step_gini_stats]).to_csv(OUTPUT_DIR / "llm_gini_step_summary_stats.csv", index=False)
print("✅ Saved step-level Gini stats summary")



df_gini.loc[len(df_gini)] = {
    "ppt_id": "ALL",
    "gini_step": overall_step,
    "gini_token": overall_token,
    "most_used_step_state": "",
    "most_used_token_state": ""
}
df_gini.to_csv(TABLE_PPT_OUTPUT, index=False)
print(f"✅ Saved Gini scores (run-level) to {TABLE_PPT_OUTPUT}")

# --- Overall dominant state proportions ---
overall_step_df = pd.Series(most_used_step_overall).value_counts(normalize=True).rename_axis("dominant_state").reset_index(name="proportion")
overall_step_df.to_csv(TABLE_OVERALL_DOMINANT, index=False)
print(f"✅ Saved overall dominant state proportions to {TABLE_OVERALL_DOMINANT}")

# --- Gini Binning and bin-wise dominant state analysis ---
bins = [0, 0.45, 0.6, 1.0]
labels = ["Balanced (≤0.45)", "Moderate (0.45–0.6)", "Unbalanced (>0.6)"]
df_gini_bin = df_gini[df_gini["ppt_id"] != "ALL"].copy()
df_gini_bin["gini_bin"] = pd.cut(df_gini_bin["gini_step"], bins=bins, labels=labels)
bin_dominant = df_gini_bin.groupby(["gini_bin", "most_used_step_state"]).size().unstack(fill_value=0)
bin_dominant = bin_dominant.div(bin_dominant.sum(axis=1), axis=0).reset_index()
bin_dominant.to_csv(TABLE_BIN_DOMINANT, index=False)
print(f"✅ Saved Gini bin dominant state breakdown to {TABLE_BIN_DOMINANT}")

# --- Task-level Gini (step) summary ---
task_gini_rows = []
for (ppt_id, task_id), group in df.groupby(["ppt_id", "task_id"]):
    freq = group["high_level_code"].value_counts()
    gini = gini_coefficient(freq.values)
    task_gini_rows.append({"ppt_id": ppt_id, "task_id": task_id, "gini_step": gini})

df_task_gini = pd.DataFrame(task_gini_rows)
task_avg = df_task_gini.groupby("task_id")["gini_step"].agg(["mean", "std", "count"]).reset_index()
task_avg.to_csv(TABLE_TASK_OUTPUT, index=False)
print(f"✅ Saved Gini summary by task to {TABLE_TASK_OUTPUT}")

# --- Boxplot (step vs token) ---
df_melt = df_gini[df_gini["ppt_id"] != "ALL"].melt(
    id_vars="ppt_id",
    value_vars=["gini_step", "gini_token"],
    var_name="Gini Type",
    value_name="value"
)
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_melt, x="Gini Type", y="value", palette="pastel", width=0.5)
sns.stripplot(data=df_melt, x="Gini Type", y="value", color="gray", alpha=0.6, jitter=0.2)
plt.title("Distribution of Gini Index Across LLM Runs")
plt.ylabel("Gini Index")
plt.tight_layout()
plt.savefig(FIGURE_GINI_BOXPLOT)
plt.close()
print("✅ Saved Gini boxplot")

# --- Lineplot: mean Gini per task ---
plt.figure(figsize=(8, 5))
sns.lineplot(data=task_avg, x=task_avg["task_id"].astype(str), y="mean", marker="o", linewidth=2)
plt.ylim(0, 1)
plt.title("Average Gini Index (Step-Based) by Task – LLM")
plt.xlabel("Task ID")
plt.ylabel("Average Gini Index")
plt.tight_layout()
plt.savefig(FIGURE_GINI_LINE)
plt.close()
print("✅ Saved Gini trend lineplot by task")

# === 📊 GINI VS ACCURACY ANALYSIS =====================================

# Load full dataset again (for accuracy)
df_full = pd.read_csv(DATA_PATH)

# Filter out "ALL" row from gini
df_gini_only = df_gini[df_gini["ppt_id"] != "ALL"].copy()

# Get accuracy per run
df_acc = df_full.groupby("ppt_id")["accuracy"].mean().reset_index()
df_acc.columns = ["ppt_id", "accuracy"]
df_acc["accuracy"] = df_acc["accuracy"].round().astype(int)


# Merge with Gini scores
df_gini_merged = df_gini_only.merge(df_acc, on="ppt_id")
df_gini_merged["accuracy_bin"] = df_gini_merged["accuracy"].map({1: "Correct", 0: "Incorrect"})

# --- 📈 Boxplot: Gini vs. Accuracy Group ---
plt.figure(figsize=(7, 5))
sns.boxplot(data=df_gini_merged, x="accuracy_bin", y="gini_step", palette="pastel")
sns.stripplot(data=df_gini_merged, x="accuracy_bin", y="gini_step", color="gray", alpha=0.5, jitter=0.2)
plt.title("Gini (Step) by Accuracy Group – LLM")
plt.ylabel("Gini Index (Step-Based)")
plt.xlabel("Accuracy")
plt.tight_layout()
plt.savefig(FIGURE_ACC_BOXPLOT)
plt.close()
print("✅ Saved Gini vs. Accuracy boxplot")

# --- 📄 Logistic Regression: Gini → Accuracy ---
model = smf.logit("accuracy ~ gini_step + gini_token", data=df_gini_merged).fit(disp=0)

with open(REGRESSION_OUT, "w") as f:
    f.write(model.summary().as_text())
print(f"✅ Saved logistic regression summary to {REGRESSION_OUT}")

# --- Add back: Histogram of Gini Coefficients ---
plt.figure(figsize=(9, 5.5))
sns.histplot(df_gini_only["gini_step"], bins=12, kde=True, color="#3182bd", edgecolor='black')
plt.xlabel("Gini Coefficient (Step-Based)")
plt.ylabel("Number of Runs")
plt.title("Distribution of Gini Coefficients (Step-Based) – LLM")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "llm_gini_histogram.png")
plt.close()
print("✅ Saved histogram: llm_gini_histogram.png")

# --- Add back: Barplot of Gini Bin Counts ---
plt.figure(figsize=(8, 5.5))
sns.countplot(data=df_gini_bin, x="gini_bin", palette="Set2", edgecolor='black')
plt.xlabel("Gini Coefficient Bin")
plt.ylabel("Number of Runs")
plt.title("Binned Distribution of Gini Coefficients – LLM")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "llm_gini_bin_barplot.png")
plt.close()
print("✅ Saved bin barplot: llm_gini_bin_barplot.png")
