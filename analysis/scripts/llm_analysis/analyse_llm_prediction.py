import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
import statsmodels.api as sm

# --- Paths ---
DATA_PATH = Path("../analysis_data/full_llm_data.csv")
RESULT_DIR = Path("../analysis_results/llm/accuracy_prediction")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_COEF = RESULT_DIR / "llm_regression_summary.csv"
TABLE_FREQ = RESULT_DIR / "llm_high_level_code_frequencies.csv"
FIGURE_COEF = RESULT_DIR / "llm_regression_coeff_barplot.png"
FIGURE_BOX = RESULT_DIR / "llm_boxplot_predictors.png"

# --- Load data ---
df = pd.read_csv(DATA_PATH)

# --- Get top 6 high-level cognitive states ---
top_codes = df["high_level_code"].value_counts().nlargest(6).index.tolist()
code_counts = pd.crosstab(df["ppt_id"], df["high_level_code"])
code_props = code_counts[top_codes].div(code_counts.sum(axis=1), axis=0)
code_props = code_props.fillna(0)  # Fill 0 for ppt_ids missing a code

# --- Save full frequency table for transparency ---
df["high_level_code"].value_counts().to_csv(TABLE_FREQ)

# --- Aggregate other predictors ---
agg = df.groupby("ppt_id").agg(
    step_count=("thought_id", "count"),
    avg_token=("token_count", "mean"),
    accuracy=("accuracy", "first")
)

# --- Combine predictors ---
X = pd.concat([agg[["step_count", "avg_token"]], code_props], axis=1)
y = agg["accuracy"]

# --- Standardize predictors for regression ---
X_std = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns).reset_index(drop=True)
y = y.reset_index(drop=True)

# --- Logistic regression with statsmodels ---
X_std_const = sm.add_constant(X_std)
model = sm.Logit(y, X_std_const)
result = model.fit(disp=False, method="bfgs")  # Use 'bfgs' to improve convergence



summary_df = result.summary2().tables[1].reset_index()
summary_df.columns = ["predictor", "coef", "std_err", "z", "p", "ci_low", "ci_high"]
summary_df.to_csv(TABLE_COEF, index=False)

# --- Coefficient plot ---
plt.figure(figsize=(8, 5))
coef_plot = summary_df[summary_df["predictor"] != "const"].sort_values("coef", ascending=False)
sns.barplot(data=coef_plot, x="coef", y="predictor", hue=None)
plt.axvline(0, color="gray", linestyle="--")
plt.title("LLM Logistic Regression Coefficients")
plt.tight_layout()
plt.savefig(FIGURE_COEF)
plt.close()

# --- Boxplot for predictors by accuracy ---
plot_df = X.copy()
plot_df["accuracy"] = y

# Drop predictors with no variance
constant_cols = plot_df.drop(columns="accuracy").nunique()[lambda x: x <= 1].index.tolist()
if constant_cols:
    print("Dropped constant predictors (zero variance):", constant_cols)
    plot_df = plot_df.drop(columns=constant_cols)

# Melt for seaborn
melted = plot_df.melt(id_vars="accuracy", var_name="predictor", value_name="value")

# Drop NaNs and filter sparse combinations
melted = melted.dropna(subset=["value", "predictor"])
group_counts = melted.groupby(["predictor", "accuracy"]).size().reset_index(name="count")
valid_pairs = group_counts[group_counts["count"] > 1][["predictor", "accuracy"]]
melted = pd.merge(melted, valid_pairs, on=["predictor", "accuracy"], how="inner")

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=melted, x="value", y="predictor", hue="accuracy", palette="Set2")
plt.title("Predictor Distributions by LLM Accuracy")
plt.tight_layout()
plt.savefig(FIGURE_BOX)
plt.close()



# --- Cross-validation accuracy ---
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])
cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
print("Cross-validated accuracy: %.3f ± %.3f" % (cv_scores.mean(), cv_scores.std()))

# --- Bootstrap validation (1000 samples) ---
boot_scores = []
for _ in range(1000):
    X_res, y_res = resample(X, y)
    pipe.fit(X_res, y_res)
    boot_scores.append(pipe.score(X, y))

boot_mean = np.mean(boot_scores)
boot_ci = np.percentile(boot_scores, [2.5, 97.5])
print(f"Bootstrapped accuracy: {boot_mean:.3f} (95% CI: {boot_ci[0]:.3f} – {boot_ci[1]:.3f})")

print("✅ LLM binary prediction complete. Outputs saved to:", RESULT_DIR)

# --- Cross-validation accuracy ---
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])
cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
cv_summary = "Cross-validated accuracy: %.3f ± %.3f" % (cv_scores.mean(), cv_scores.std())
print(cv_summary)
with open(RESULT_DIR / "llm_cv_accuracy.txt", "w") as f:
    f.write(cv_summary + "\n")

# --- Bootstrap validation (1000 samples) ---
boot_scores = []
for _ in range(1000):
    X_res, y_res = resample(X, y)
    pipe.fit(X_res, y_res)
    boot_scores.append(pipe.score(X, y))

boot_mean = np.mean(boot_scores)
boot_ci = np.percentile(boot_scores, [2.5, 97.5])
boot_summary = f"Bootstrapped accuracy: {boot_mean:.3f} (95% CI: {boot_ci[0]:.3f} – {boot_ci[1]:.3f})"
print(boot_summary)
with open(RESULT_DIR / "llm_bootstrap_accuracy.txt", "w") as f:
    f.write(boot_summary + "\n")

print("✅ LLM binary prediction complete. Outputs saved to:", RESULT_DIR)
