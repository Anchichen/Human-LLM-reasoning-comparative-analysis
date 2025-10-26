import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# === PATH SETUP ===
DATA_HUMAN = Path("../analysis_data/full_human_data.csv")
DATA_LLM = Path("../analysis_data/full_llm_data_subset42.csv")
OUTPUT_DIR = Path("../analysis_results/compare/accuracy_prediction")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === COLUMN SANITIZATION FUNCTION ===
def sanitize_column_names(df):
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        clean_col = re.sub(r'\W|^(?=\d)', '_', col)
        rename_map[col] = clean_col
    return df.rename(columns=rename_map), rename_map

# === PREPROCESS FUNCTION ===
def preprocess(data, group_label):
    data = data.copy()
    data['group'] = group_label
    grouped = data.groupby(['ppt_id', 'task_id'])
    
    features = grouped.agg(
        step_count=('thought_id', 'count'),
        avg_token_per_thought=('token_count', 'mean'),
        transition_entropy=('high_level_code', lambda x: -np.sum((x.value_counts(normalize=True) * np.log2(x.value_counts(normalize=True) + 1e-9)))),
        transition_gini=('high_level_code', lambda x: 1 - np.sum((x.value_counts(normalize=True) ** 2)))
    ).reset_index()

    # Proportion of states
    state_counts = data.groupby(['ppt_id', 'task_id', 'high_level_code']).size().unstack(fill_value=0)
    state_props = state_counts.div(state_counts.sum(axis=1), axis=0).reset_index()

    # Accuracy
    acc = data.groupby(['ppt_id', 'task_id'])['accuracy'].first().reset_index()

    merged = pd.merge(features, state_props, on=['ppt_id', 'task_id'])
    merged = pd.merge(merged, acc, on=['ppt_id', 'task_id'])
    merged['group'] = group_label
    return merged

# === LOAD DATA ===
human_data = pd.read_csv(DATA_HUMAN)
llm_data = pd.read_csv(DATA_LLM)

human_processed = preprocess(human_data, 'Human')
llm_processed = preprocess(llm_data, 'LLM')
combined_data = pd.concat([human_processed, llm_processed], ignore_index=True).dropna()

# === USE REDUCED PREDICTORS TO AVOID MULTICOLLINEARITY ===
top_predictors = ['step_count', 'avg_token_per_thought', 'transition_entropy', 'transition_gini',
                  'Planning', 'Hypothesis', 'Evaluation/Monitoring', 'DecisionMaking']

# === LOGISTIC REGRESSION PER GROUP ===
def run_logit(data, group_name, custom_predictors=None):
    df, rename_map = sanitize_column_names(data)
    predictors = custom_predictors if custom_predictors else top_predictors
    valid_cols = [col for col in predictors if col in rename_map]
    predictors_renamed = [rename_map[col] for col in valid_cols]
    formula = 'accuracy ~ ' + ' + '.join(predictors_renamed)
    model = smf.logit(formula, data=df).fit(disp=0)
    summary = model.summary2().tables[1]
    summary['odds_ratio'] = np.exp(summary['Coef.'])
    summary['group'] = group_name
    summary['predictor'] = summary.index
    return summary.reset_index(drop=True)


# === INTERACTION MODEL ===
def run_interaction_model(data):
    df, rename_map = sanitize_column_names(data)
    df['group_binary'] = (df['group'] == 'LLM').astype(int)

    valid_cols = []
    for col in top_predictors:
        if col in rename_map:
            safe_col = rename_map[col]
            values_human = df[df['group_binary'] == 0][safe_col]
            values_llm = df[df['group_binary'] == 1][safe_col]
            if values_human.nunique() > 1 and values_llm.nunique() > 1:
                valid_cols.append(col)

    dropped = [col for col in top_predictors if col in rename_map and col not in valid_cols]
    print("🔍 Dropped interaction terms due to zero variance in one group:", dropped)

    if not valid_cols:
        print("⚠️ No valid predictors left for interaction model. Skipping.")
        return pd.DataFrame(columns=['predictor', 'Coef.', 'Std.Err.', 'z', 'P>|z|', 'odds_ratio', 'group'])

    predictors_renamed = [rename_map[col] for col in valid_cols]
    interaction_terms = [f'{col}*group_binary' for col in predictors_renamed]

    formula = 'accuracy ~ group_binary + ' + ' + '.join(predictors_renamed + interaction_terms)
    model = smf.logit(formula, data=df).fit(disp=0)

    summary = model.summary2().tables[1]
    summary['odds_ratio'] = np.exp(summary['Coef.'])
    summary['group'] = 'Combined'
    summary['predictor'] = summary.index
    return summary.reset_index(drop=True)



# === RUN MODELS ===
human_predictors = top_predictors + ['Reflection']
summary_human = run_logit(human_processed, 'Human', custom_predictors=human_predictors)

summary_llm = run_logit(llm_processed, 'LLM')
summary_combined = run_interaction_model(combined_data)

# === SAVE TABLES ===
summary_all = pd.concat([summary_human, summary_llm, summary_combined], ignore_index=True)

# Reorder columns: group and predictor first
ordered_cols = ['group', 'predictor'] + [col for col in summary_all.columns if col not in ['group', 'predictor']]
summary_all = summary_all[ordered_cols]

# Add significance marker column based on p-values
def mark_significance(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

summary_all['significance'] = summary_all['P>|z|'].apply(mark_significance)
summary_all.to_csv(OUTPUT_DIR / "accuracy_prediction_coefficients.csv", index=False)

# === PLOT COEFFICIENT BARPLOT ===
coef_plot_data = summary_all[(summary_all['group'] != 'Combined') & (summary_all['predictor'] != 'Intercept')]

plt.figure(figsize=(12, 6))
ax = sns.barplot(data=coef_plot_data, x='predictor', y='Coef.', hue='group')

plt.axhline(0, color='black', linestyle='--')
plt.xticks(rotation=45, ha='right')
plt.title("Logistic Regression Coefficients by Group")

# === Annotate Top 2 Human Significant Predictors ===
# Filter to human + p < 0.05, sort by abs(Coef)
top_sig_human = (
    coef_plot_data[(coef_plot_data['group'] == 'Human') & (coef_plot_data['P>|z|'] < 0.05)]
    .copy()
    .sort_values(by='Coef.', key=lambda x: x.abs(), ascending=False)
    .head(2)
)

for i, row in coef_plot_data.iterrows():
    if row['group'] == 'Human' and row['P>|z|'] < 0.05:
        predictor = row['predictor']
        coef_val = row['Coef.']

        for bar in ax.patches:
            bar_center = bar.get_x() + bar.get_width() / 2
            bar_label_index = int(round(bar_center))
            xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
            
            if 0 <= bar_label_index < len(xtick_labels):
                bar_label = xtick_labels[bar_label_index]
                if bar_label == predictor:
                    y = bar.get_height()
                    y_pos = y + 0.05 if y > 0 else y - 0.05
                    va = 'bottom' if y > 0 else 'top'
                    ax.text(bar_center, y_pos, '*', ha='center', va=va, color='black', fontsize=15, fontweight='bold')
                    break

# Ensure y-axis has enough space for markers
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax + 0.05)


plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_prediction_coef_barplot.png")
plt.close()




# === PLOT INTERACTIONS ===
def plot_interaction(data, predictor):
    plt.figure(figsize=(6, 4))
    sns.regplot(data=data[data['group'] == 'Human'], x=predictor, y='accuracy', logistic=True, label='Human', scatter=False)
    sns.regplot(data=data[data['group'] == 'LLM'], x=predictor, y='accuracy', logistic=True, label='LLM', scatter=False)
    plt.title(f"Interaction: {predictor} × Group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"interaction_plot_{predictor}.png")
    plt.close()

for var in ['step_count', 'transition_entropy']:
    if var in combined_data.columns:
        plot_interaction(combined_data, var)

print("✅ Accuracy prediction analysis completed. Results saved to:", OUTPUT_DIR)
