import os
import json
import time
import pandas as pd
from pathlib import Path
from collections import defaultdict
import tiktoken
from openai import OpenAI

# === Project Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLM_CODED_PATH = PROJECT_ROOT / "llm_auto_coding" / "coded_llm_raterB" / "by_task_raterB"
LLM_SCORE_PATH = PROJECT_ROOT / "analysis" / "analysis_data" / "llm_final_guesses_scored.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "analysis" / "analysis_data" / "full_llm_data_raterB.csv"

# === Tokenizer Setup ===
tokenizer = tiktoken.get_encoding("cl100k_base")
def count_tokens(text):
    return len(tokenizer.encode(text))

# === OpenAI client ===
client = OpenAI()

# === Codebook: low-level to high-level ===
LOW_TO_HIGH_MAP = {
    "readingInstructions": "Orientation",
    "misinterpretingInstructions": "Orientation",
    "planningUnorganised": "Planning",
    "planningExplorative": "Planning",
    "planningBasicComparison": "Planning",
    "planningStrategic": "Planning",
    "adherePlan": "changePlan",
    "modifyPlan": "changePlan",
    "processingGlobal": "ProcessingScope",
    "processingLocal": "ProcessingScope",
    "representationGist": "MentalRepresentation",
    "representationVerbatim": "MentalRepresentation",
    "hypoGeneration": "Hypothesis",
    "hypoRevision": "Hypothesis",
    "hypoAlternative": "Hypothesis",
    "monitorHighLevel": "Evaluation/Monitoring",
    "monitorLowLevel": "Evaluation/Monitoring",
    "ruleExclusion": "Evaluation/Monitoring",
    "counterExampleSearch": "Evaluation/Monitoring",
    "decisionConfirm": "DecisionMaking",
    "decisionReject": "DecisionMaking",
    "reflectTask": "Reflection",
    "reflectSelf": "Reflection",
    "reflectUncertainty": "Reflection",
    "ruleArticulation": "FinalRule",
    "ruleGuess": "FinalRule",
    "ruleImpasse": "FinalRule",
    "memoryLoss": "Memory",
    "memoryRegain": "Memory",
    "memoryFalseRegain": "Memory"
}

INFERRED_HIGH_LEVEL = {
    "Orientation": "Orientation",
    "planning": "Planning",
    "processing": "ProcessingScope",
    "Inference": "Hypothesis",
    "Evaluation": "Evaluation/Monitoring",
    "Reflection": "Reflection",
    "FinalRule": "FinalRule"
}

# === OpenAI fallback ===
def suggest_high_level_code_openai(low_code, text, temperature=0):
    system_prompt = """You are assisting with cognitive coding of think-aloud protocols from a Zendo reasoning task experiment.

In each session, a llm participant tries to discover a hidden rule by observing sets of visual panels (some with stars, some without), forming hypotheses, and testing those hypotheses against evidence. Their verbal thoughts have been segmented and lightly labeled with low-level tags.

Your job is to assign the **best-fitting high-level cognitive category** to each thought, based on the low-level label and the text of the thought.

Here are the official cognitive state categories (high-level codes):

1. Orientation – Early framing of the task or reorientation to instructions.
2. Planning – Laying out strategies or comparisons to pursue.
3. changePlan – Stating that a plan is being changed or continued.
4. ProcessingScope – Zooming in (local) or scanning across (global) visual features.
5. MentalRepresentation – Mentions of abstracted or verbatim representations.
6. Hypothesis – Generating or revising possible rules.
7. Evaluation/Monitoring – Checking, validating, or seeking disconfirming cases.
8. DecisionMaking – Making a binary choice about rule success/failure.
9. Reflection – Commenting on one’s own performance or confidence.
10. FinalRule – Final articulation of a rule or state of impasse.
11. Memory – Refers to forgetting, recalling, or misrecalling something."""

    user_prompt = f"""Low-level label: {low_code}
Thought text: {text}

Which high-level category best fits this? Respond with just the label."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ OpenAI error for: {text[:50]}... → {e}")
        return None

# === Load LLM coded JSON ===
def load_llm_thought_data():
    all_data = []
    for file in sorted(LLM_CODED_PATH.glob("task*_coded_llm_raterB.json")):
        with open(file, "r") as f:
            all_runs = json.load(f)

        for run_block in all_runs:
            run_full = run_block["run_id"]  # e.g., "run01_task1"
            run_parts = run_full.replace("run", "").split("_task")
            run_id = int(run_parts[0].lstrip("0"))  # ensure match with Excel
            task_id = int(run_parts[1].lstrip("0"))

            for step in run_block["coded_steps"]:
                text = step.get("span", "")
                low_code = step.get("open_code", "").strip()
                high_code = LOW_TO_HIGH_MAP.get(low_code)
                flag = "direct"

                if high_code is None:
                    if low_code in INFERRED_HIGH_LEVEL:
                        high_code = INFERRED_HIGH_LEVEL[low_code]
                        flag = "inferred"
                    elif any(low_code.startswith(k.lower()) for k in INFERRED_HIGH_LEVEL):
                        key = next(k for k in INFERRED_HIGH_LEVEL if low_code.startswith(k.lower()))
                        high_code = INFERRED_HIGH_LEVEL[key]
                        flag = "inferred_partial"
                    else:
                        flag = "unmapped"
                        high_code = None

                all_data.append({
                    "task_id": task_id,
                    "run_id": run_id,
                    "thought_id": step.get("id", ""),
                    "text": text,
                    "low_level_code": low_code,
                    "high_level_code": high_code,
                    "token_count": count_tokens(text),
                    "flag": flag
                })

    return pd.DataFrame(all_data)

# === Load accuracy ===
def load_llm_scores():
    df = pd.read_excel(LLM_SCORE_PATH)
    return df[["task_id", "run_id", "accuracy"]]

# === Merge and clean ===
def merge_llm_data(thought_df, score_df):
    return thought_df.merge(score_df, on=["task_id", "run_id"], how="left")

# === Master function ===
def load_all_llm_data(save_csv=True):
    print("📥 Loading thoughts...")
    df_thoughts = load_llm_thought_data()
    print(f"✅ {len(df_thoughts)} thoughts loaded.")

    print("📥 Loading scores...")
    df_scores = load_llm_scores()
    print(f"✅ {len(df_scores)} accuracy rows loaded.")

    df_merged = merge_llm_data(df_thoughts, df_scores)
    print(f"🔗 Merged: {df_merged.shape[0]} rows")

    # OpenAI fallback for unmapped
    unmapped_rows = df_merged[df_merged["flag"] == "unmapped"]
    if len(unmapped_rows) > 0:
        print(f"🔮 Calling OpenAI for {len(unmapped_rows)} unmapped rows...")
        for idx, row in unmapped_rows.iterrows():
            label = suggest_high_level_code_openai(row["low_level_code"], row["text"])
            if label:
                df_merged.at[idx, "high_level_code"] = label
                df_merged.at[idx, "flag"] = "unmapped_ai_assigned"
            time.sleep(1.2)
        print("✅ Auto-labeling completed.")

    # Final sorting and saving
    df_merged = df_merged.sort_values(by=["task_id", "run_id", "thought_id"])
    column_order = [
        "task_id", "run_id", "thought_id", "text",
        "high_level_code", "low_level_code", "token_count",
        "flag", "accuracy"
    ]
    df_merged = df_merged[column_order]

    if save_csv:
        df_merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
        print(f"💾 Saved to {OUTPUT_PATH}")

    return df_merged

# === Execute ===
if __name__ == "__main__":
    df_llm = load_all_llm_data()
    print(df_llm.head())
