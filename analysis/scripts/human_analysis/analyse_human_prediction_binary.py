import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pathlib import Path
from tqdm import tqdm

# --- Paths ---
DATA_PATH = Path("../analysis_data/full_human_data.csv")
FIGURE_DIR = Path("../analysis_results/human/figures")
TABLE_DIR = Path("../analysis_results/human/tables")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# --- Load data ---
df = pd.read_csv(DATA_PATH)
df["reaction"] = df["reaction"].str.strip()

# --- Aggregate per participant ---
grouped = df.groupby("ppt_id")
features_df = grouped.agg({
    "token_count": "sum",
    "accuracy": "mean",
    "thought_id": "count"
}).rename(columns={"thought_id": "step_count"})

top_states = ["ProcessingScope", "Hypothesis", "Reflection", "DecisionMaking"]
for state in top_states:
    count = df[df["high_level_code"] == state].groupby("ppt_id").size()
    features_df[state] = features_df.index.map(count).fillna(0)

features_df["avg_token_per_thought"] = features_df["token_count"] / features_df["step_count"]
for state in top_states:
    features_df[state] = features_df[state] / features_df["step_count"]
features_df["accuracy_bin"] = features_df["accuracy"].round().astype(int)

# --- Logistic regression model ---
formula = "accuracy_bin ~ step_count + avg_token_per_thought + " + " + ".join(top_states)
model = smf.logit(formula=formula, data=features_df).fit(disp=0)

# --- Save regression summary ---
summary_path = TABLE_DIR / "human_accuracy_binary_model_summary.txt"
with open(summary_path, "w") as f:
    f.write(model.summary().as_text())
print(f"✅ Saved model summary to {summary_path}")

# --- Coefficient plot ---
coef = model.params.drop("Intercept")
conf = model.conf_int().drop("Intercept")
sorted_coef = coef.sort_values()
conf_sorted = conf.loc[sorted_coef.index]

plt.figure(figsize=(8, 6))
sns.barplot(x=sorted_coef.values, y=sorted_coef.index, orient="h", palette="viridis")
plt.errorbar(x=sorted_coef.values, y=np.arange(len(sorted_coef)),
             xerr=(sorted_coef - conf_sorted[0], conf_sorted[1] - sorted_coef),
             fmt='none', c='black', capsize=3)
plt.axvline(0, linestyle="--", color="gray")
plt.title("Coefficient Estimates (Binary Accuracy Model)")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "human_accuracy_binary_coefplot.png")
plt.close()
print("✅ Saved coefficient plot")

# --- Boxplots ---
for y in ["step_count", "avg_token_per_thought"]:
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=features_df, x="accuracy_bin", y=y, palette="pastel")
    sns.stripplot(data=features_df, x="accuracy_bin", y=y, color="gray", alpha=0.5, jitter=0.2)
    plt.title(f"{y.replace('_', ' ').title()} by Accuracy")
    plt.xticks([0, 1], ["Incorrect", "Correct"])
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"human_accuracy_binary_{y}_boxplot.png")
    plt.close()
    print(f"✅ Saved boxplot for {y}")

# --- Cross-validation (5-fold) ---
X_cols = ["step_count", "avg_token_per_thought"] + top_states
X = features_df[X_cols].values
y = features_df["accuracy_bin"].values
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accs = []
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    accs.append(acc)

with open(TABLE_DIR / "human_accuracy_crossval_summary.txt", "w") as f:
    f.write("Cross-Validation Accuracy Scores:\n")
    for i, a in enumerate(accs):
        f.write(f"  Fold {i+1}: {a:.3f}\n")
    f.write(f"\nAverage Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
print("✅ Saved cross-validation results")

# --- Bootstrap validation ---
boot_results = []
for _ in tqdm(range(1000), desc="🔁 Bootstrapping"):
    sample = features_df.sample(n=len(features_df), replace=True)
    try:
        m = smf.logit(formula=formula, data=sample).fit(disp=0)
        boot_results.append(m.params)
    except:
        continue

boot_df = pd.DataFrame(boot_results)
boot_df.to_csv(TABLE_DIR / "human_accuracy_bootstrap_coefficients.csv", index=False)

# --- Bootstrap coefficient plot ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=boot_df[top_states + ["step_count", "avg_token_per_thought"]], orient="h", palette="Set3")
plt.axvline(0, linestyle="--", color="gray")
plt.title("Bootstrapped Coefficient Distributions (Accuracy Model)")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "human_accuracy_bootstrap_coefplot.png")
plt.close()
print("✅ Saved bootstrap coefficient plot")
