# === Setup ===
library(tidyverse)
library(MASS)       # For stepAIC
library(broom)      # For tidy output
library(lme4)       # For mixed-effects model

# === Paths ===
setwd("/Users/anchichen/Desktop/LLMDesignConfig")
data_path <- "analysis/analysis_data/full_llm_data_subset42.csv"
output_dir <- "analysis/analysis_results/compare/stepwise_prediction_llm"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(42)

# === Load and clean data ===
df <- read_csv(data_path)

# === Calculate per-ppt-task-level features ===
df_grouped <- df %>%
  group_by(ppt_id, task_id) %>%
  summarise(
    step_count = n(),
    avg_token_per_thought = mean(token_count, na.rm = TRUE),
    accuracy = first(accuracy),
    .groups = "drop"
  )

# === Calculate high-level code usage proportion per ppt_id × task_id ===
df_states <- df %>%
  count(ppt_id, task_id, high_level_code) %>%
  group_by(ppt_id, task_id) %>%
  mutate(prop = n / sum(n)) %>%
  pivot_wider(names_from = high_level_code, values_from = prop, values_fill = 0)

# Explicitly drop 'n' column to prevent accidental use as predictor
df_states$n <- NULL

# === Calculate number of unique high-level codes per ppt-task (cognitive diversity) ===
df_diversity <- df %>%
  group_by(ppt_id, task_id) %>%
  summarise(n_states = n_distinct(high_level_code), .groups = "drop")


# === Merge features and clean column names ===
df_merged <- df_grouped %>%
  left_join(df_states, by = c("ppt_id", "task_id")) %>%
  left_join(df_diversity, by = c("ppt_id", "task_id"))

# Clean column names
names(df_merged) <- gsub("/", "_", names(df_merged))
names(df_merged) <- gsub(" ", "_", names(df_merged))

# === Standardize numeric predictors (compatible with older dplyr) ===
predictors <- setdiff(names(df_merged), c("ppt_id", "task_id", "accuracy"))

# Identify numeric columns manually
numeric_cols <- names(df_merged)[sapply(df_merged, is.numeric)]
numeric_cols <- intersect(numeric_cols, predictors)

# Scale only those columns
df_merged[numeric_cols] <- scale(df_merged[numeric_cols])


# === Step 1: Stepwise Regression using GLM (no random effect) ===
formula_glm <- as.formula(paste("accuracy ~", paste(predictors, collapse = " + ")))
fit_full <- glm(formula_glm, data = df_merged, family = binomial)
# === Run stepwise logistic regression and capture variable actions ===
step_trace_txt <- file.path(output_dir, "stepwise_trace_with_variables.txt")
sink(step_trace_txt)  # Start capturing console output
fit_step <- stepAIC(fit_full, direction = "both", trace = TRUE)
sink()  # Stop capturing

# === Output structured AIC trace table ===
step_trace <- as.data.frame(fit_step$anova)

# Safely add rownames as a new column called "Action"
if (!"Action" %in% names(step_trace)) {
  step_trace <- step_trace %>%
    tibble::rownames_to_column(var = "Action")
}

# Optional: reorder columns to make it readable
step_trace <- step_trace %>%
  dplyr::select(Action, everything())

write_csv(step_trace, file.path(output_dir, "stepwise_aic_trace_llm.csv"))

fit_metrics_glm <- broom::glance(fit_step)
write_csv(fit_metrics_glm, file.path(output_dir, "fit_metrics_stepwise_glm.csv"))



# Save stepwise-selected predictors
selected_predictors <- names(coef(fit_step))[-1]  # remove intercept

# === Output GLM summary ===
summary_glm <- broom::tidy(fit_step) %>%
  mutate(signif = ifelse(p.value < 0.05, "*", "")) %>%
  dplyr::rename(
    predictor = term,
    coefficient = estimate,
    std_error = std.error,
    z_score = statistic,
    p_value = p.value
  ) %>%
  dplyr::select(predictor, coefficient, std_error, z_score, p_value, signif)

write_csv(summary_glm, file.path(output_dir, "stepwise_logit_summary_llm.csv"))
cat("✅ Stepwise GLM complete. Summary saved.\n")

# === Step 2: Run GLMER using selected predictors ===
if (length(selected_predictors) > 0) {
  formula_glmer <- as.formula(paste("accuracy ~", paste(selected_predictors, collapse = " + "), "+ (1 | ppt_id)", "+ (1 | task_id)"))
  fit_glmer <- glmer(
    formula_glmer, data = df_merged, family = binomial,
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  )
  
  if (!is.null(fit_glmer@optinfo$conv$lme4$messages)) {
    warning("⚠️ Convergence warnings:\n", paste(fit_glmer@optinfo$conv$lme4$messages, collapse = "\n"))
  }
  
  summary_glmer <- broom.mixed::tidy(fit_glmer) %>%
    filter(effect == "fixed") %>%
    mutate(signif = ifelse(p.value < 0.05, "*", "")) %>%
    dplyr::rename(
      predictor = term,
      coefficient = estimate,
      std_error = std.error,
      z_score = statistic,
      p_value = p.value
    ) %>%
    dplyr::select(predictor, coefficient, std_error, z_score, p_value, signif)
  
  write_csv(summary_glmer, file.path(output_dir, "stepwise_mixed_logit_summary_llm.csv"))
  cat("✅ GLMER with selected predictors complete. Summary saved.\n")
} else {
  cat("⚠️ No predictors were selected in stepwise regression. GLMER skipped.\n")
}

fit_metrics_glmer <- broom.mixed::glance(fit_glmer)
write_csv(fit_metrics_glmer, file.path(output_dir, "fit_metrics_glmer.csv"))

