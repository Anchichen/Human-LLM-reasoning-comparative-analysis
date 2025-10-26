library(tidyverse)

# === Setup ===
output_dir <- "analysis/analysis_results/compare/accuracy_prediction"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Load GLMER summaries
glmer_llm <- read_csv(file.path(output_dir, "stepwise_prediction_llm/stepwise_mixed_logit_summary_llm.csv")) %>%
  mutate(group = "LLM")

glmer_human <- read_csv(file.path(output_dir, "stepwise_prediction_human/stepwise_mixed_logit_summary_human.csv")) %>%
  mutate(group = "Human")

# Combine and ensure full predictor list
df_plot <- bind_rows(glmer_llm, glmer_human) %>%
  complete(predictor, group, fill = list(z_score = NA, p_value = NA, signif = ""))

# === Exclude unwanted predictors ===
df_plot <- df_plot %>%
  filter(!predictor %in% c("(Intercept)", "changePlan", "DecisionMaking"))

# Sort by importance (mean abs z-score)
predictor_order <- df_plot %>%
  group_by(predictor) %>%
  summarise(mean_z = mean(abs(z_score), na.rm = TRUE)) %>%
  arrange(desc(mean_z)) %>%
  pull(predictor)

df_plot$predictor <- factor(df_plot$predictor, levels = predictor_order)

# Optional: Prettify x-axis labels with line breaks
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
  "transition_gini" = "Transition\nGini"
)

# Assign and order pretty labels
df_plot$pretty_predictor <- pretty_labels[as.character(df_plot$predictor)]
df_plot$pretty_predictor <- factor(df_plot$pretty_predictor,
                                   levels = pretty_labels[predictor_order])



# === Plot ===
p <- ggplot(df_plot, aes(x = pretty_predictor, y = z_score, fill = group)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.65, na.rm = TRUE) +
  geom_text(
    aes(label = ifelse(p_value < 0.05, "*", "")),
    position = position_dodge(width = 0.8),
    vjust = -0.6,
    size = 7,
    na.rm = TRUE
  ) +
  scale_fill_manual(
    values = c("Human" = "#4E79A7", "LLM" = "#F28E2B"),
    labels = c("Human", "LLM")
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    title = "Z-scores of Predictors from GLMER (Human vs LLM)",
    x = NULL,
    y = "Z-score",
    fill = "Group"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5, size = 13),
    axis.text.y = element_text(size = 13),
    axis.title.y = element_text(size = 14, face = "bold", margin = margin(r = 12)),  # move y-axis label farther
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    legend.position = c(0.98, 0.05),  # bottom right inside plot
    legend.justification = c("right", "bottom"),
    legend.background = element_rect(fill = alpha("white", 0.8), color = NA),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 11),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = margin(t = 10, r = 15, b = 10, l = 15)
  )


# === Save ===
ggsave(
  filename = file.path(output_dir, "glmer_zscore_comparison.png"),
  plot = p,
  width = 10,
  height = 5.5,
  dpi = 300
)

print(p)

