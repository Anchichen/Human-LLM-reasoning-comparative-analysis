library(tidyverse)

# === Setup ===
output_dir <- "../analysis_results/compare/accuracy_prediction"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# -------------------------------
# Load Step 2 (GLMER) summaries
# Expect: predictor, coefficient, std_error, z_score, p_value, signif
# -------------------------------
glmer_llm <- read_csv(file.path(output_dir, "stepwise_prediction_llm/stepwise_mixed_logit_summary_llm.csv")) %>%
  mutate(group = "LLM")

glmer_human <- read_csv(file.path(output_dir, "stepwise_prediction_human/stepwise_mixed_logit_summary_human.csv")) %>%
  mutate(group = "Human")

glmer_combined <- bind_rows(glmer_llm, glmer_human) %>%
  rename(
    p_step2   = p_value,
    beta_step2 = coefficient,
    std_error = std_error
  )

# -------------------------------
# Load Step 1 (GLM) summaries (for markers)
# Expect: predictor, p_value (and optionally coefficient)
# -------------------------------
glm_llm <- read_csv(file.path(output_dir, "stepwise_prediction_llm/stepwise_logit_summary_llm.csv")) %>%
  mutate(group = "LLM") %>%
  select(predictor, group, p_value) %>%
  rename(p_step1 = p_value)

glm_human <- read_csv(file.path(output_dir, "stepwise_prediction_human/stepwise_logit_summary_human.csv")) %>%
  mutate(group = "Human") %>%
  select(predictor, group, p_value) %>%
  rename(p_step1 = p_value)

glm_combined <- bind_rows(glm_llm, glm_human)

# -------------------------------
# Combine Step 1 + Step 2
# -------------------------------
df_plot <- glmer_combined %>%
  full_join(glm_combined, by = c("predictor", "group")) %>%
  complete(predictor, group,
           fill = list(
             beta_step2 = NA_real_,
             std_error  = NA_real_,
             z_score    = NA_real_,
             p_step2    = NA_real_,
             p_step1    = NA_real_,
             signif     = ""
           )) %>%
  filter(predictor != "(Intercept)")

# -------------------------------
# Keep predictors significant in Step 1 OR Step 2 for at least one group
# -------------------------------
keep_always <- c("n_states")

df_filtered <- df_plot %>%
  filter(
    !is.na(predictor) &
      (predictor %in% keep_always |
         coalesce(p_step1, 1) < 0.05 |
         coalesce(p_step2, 1) < 0.05)
  )

# Ensure both groups appear for each kept predictor (draw zero bar if missing)
df_filtered <- df_filtered %>%
  complete(predictor, group,
           fill = list(beta_step2 = NA_real_, std_error = NA_real_,
                       z_score = NA_real_, p_step2 = NA_real_,
                       p_step1 = NA_real_, signif = "")) %>%
  mutate(
    in_step2 = !is.na(beta_step2),
    beta_step2_plot = coalesce(beta_step2, 0)  # use this for y aesthetic
  )

# -------------------------------
# Pretty labels
# -------------------------------
pretty_labels <- c(
  "step_count" = "Step Count",
  "avg_token_per_thought" = "Avg Token\nper Thought",
  "n_states" = "Num. of\nStates",
  "Reflection" = "Reflection",
  "Memory" = "Memory",
  "Planning" = "Planning",
  "Hypothesis" = "Hypothesis",
  "Evaluation_Monitoring" = "Evaluation\nMonitoring",
  "transition_entropy" = "Transition\nEntropy",
  "transition_gini" = "Transition\nGini",
  "DecisionMaking" = "Decision\nMaking",
  "changePlan" = "Change\nPlan"
)

df_filtered <- df_filtered %>%
  mutate(
    pretty_predictor = if_else(
      predictor %in% names(pretty_labels),
      pretty_labels[as.character(predictor)],
      as.character(predictor)
    )
  ) %>%
  filter(!is.na(pretty_predictor))

# -------------------------------
# Ordering (by mean |β| across groups)
# -------------------------------
predictor_order <- df_filtered %>%
  group_by(predictor) %>%
  summarise(mean_beta = mean(abs(beta_step2), na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(mean_beta)) %>%
  pull(predictor)

pretty_levels <- sapply(predictor_order, function(x) {
  if (!is.na(pretty_labels[x])) pretty_labels[x] else x
}) %>% as.character()

df_filtered$predictor <- factor(df_filtered$predictor, levels = predictor_order)
df_filtered$pretty_predictor <- factor(df_filtered$pretty_predictor, levels = pretty_levels)

# -------------------------------
# Significance markers
# -------------------------------
df_filtered <- df_filtered %>%
  mutate(
    sig_step1 = case_when(
      is.na(p_step1) ~ "",
      p_step1 < 0.05 ~ "†",
      TRUE ~ ""
    ),
    sig_step2 = case_when(
      is.na(p_step2) ~ "",
      p_step2 < 0.05 ~ "*",
      TRUE ~ ""
    )
  )

# -------------------------------
# Plot (bars = Step 2 β; markers = Step 1 dagger, Step 2 asterisk)
# -------------------------------
dodge <- position_dodge(width = 0.8)
marker_offset <- 0.04  # close to the bar

p <- ggplot(df_filtered, aes(x = pretty_predictor, y = beta_step2, fill = group)) +
  geom_col(position = dodge, width = 0.65, colour = "black", linewidth = 0.2, na.rm = TRUE) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  
  # Step 1 marker (dagger) – close to bar
  geom_text(
    aes(
      label = sig_step1,
      y = beta_step2 + sign(replace_na(beta_step2, 0)) * marker_offset
    ),
    position = dodge,
    vjust = ifelse(replace_na(df_filtered$beta_step2, 0) >= 0, -0.2, 1.2),
    family = "Arial",
    size = 6,
    fontface = "bold",
    na.rm = TRUE
  ) +
  # Step 2 marker (asterisk) – slightly above Step 1 marker
  geom_text(
    aes(
      label = sig_step2,
      y = beta_step2 + sign(replace_na(beta_step2, 0)) * (marker_offset * 3)
    ),
    position = dodge,
    vjust = ifelse(replace_na(df_filtered$beta_step2, 0) >= 0, -0.2, 1.2),
    family = "Arial",
    size = 6,
    fontface = "bold",
    na.rm = TRUE
  ) +
  
  scale_fill_manual(values = c("Human" = "#4E79A7", "LLM" = "#F28E2B"), labels = c("Human", "LLM")) +
  labs(
    title = "Significant Predictors of Accuracy from Step 1 or Step 2 Models",
    x = NULL,
    y = "Standardised Coefficient (β)",
    fill = "Group"
  ) +
  theme_minimal(base_size = 16, base_family = "Arial") +
  theme(
    axis.text.x  = element_text(angle = 0, hjust = 0.5, size = 14),
    axis.text.y  = element_text(size = 14),
    axis.title.y = element_text(size = 16, face = "bold", margin = margin(r = 12)),
    plot.title   = element_text(size = 18, face = "bold", hjust = 0.5),
    legend.position = c(0.98, 0.05),
    legend.justification = c("right", "bottom"),
    legend.background = element_rect(fill = alpha("white", 0.8), color = NA),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank()
  )

# === Save ===
ggsave(file.path(output_dir, "filtered_step1or2_sig_plot.png"), p,
       width = 10, height = 5.5, dpi = 300)

print(p)


