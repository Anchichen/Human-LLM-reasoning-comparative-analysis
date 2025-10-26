import os
import json
import time
import pandas as pd
from pathlib import Path
from collections import defaultdict
import tiktoken
from openai import OpenAI

# === Project Paths ===
HUMAN_CODED_PATH = Path("runs/coded_human")
HUMAN_SCORE_PATH = Path("analysis/analysis_data/human_final_guesses_scored.xlsx")
OUTPUT_PATH = Path("analysis/analysis_data/full_human_data.csv")

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

# === OpenAI-based suggestion for unmapped codes ===
def suggest_high_level_code_openai(low_code, text, temperature=0):
    system_prompt = """You are assisting with cognitive coding of think-aloud protocols from a Zendo reasoning task experiment.

In each session, a human participant tries to discover a hidden rule by observing sets of visual panels (some with stars, some without), forming hypotheses, and testing those hypotheses against evidence. Their verbal thoughts have been segmented and lightly labeled with low-level tags.

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
11. Memory – Refers to forgetting, recalling, or misrecalling something.

You must return ONLY the name of the most appropriate high-level code from the list above. Do not invent new categories."""

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

# === Load human thought data ===
def load_human_thought_data():
    all_data = []
    for file in HUMAN_CODED_PATH.glob("p*_coded.json"):
        ppt_id = file.stem.split("_")[0]
        with open(file, "r") as f:
            full_json = json.load(f)

        for task_block in full_json:
            title_block = task_block[0]
            task_id_str = title_block.get("Title", "").lower().replace("scene", "")
            try:
                task_id = int(task_id_str)
            except ValueError:
                print(f"⚠️ Invalid task title: {title_block}")
                continue

            for t in task_block[1:]:
                text = t.get("span", "")
                low_code = t.get("open_code", "").strip()
                high_code = LOW_TO_HIGH_MAP.get(low_code)
                flag = "direct"

                if high_code is None:
                    if low_code in INFERRED_HIGH_LEVEL:
                        high_code = INFERRED_HIGH_LEVEL[low_code]
                        flag = "inferred"
                        print(f"⚠️ Inferred: '{low_code}' → '{high_code}' in {ppt_id}, task {task_id}, thought {t.get('id')}")
                    elif any(low_code.startswith(k.lower()) for k in INFERRED_HIGH_LEVEL):
                        key = next(k for k in INFERRED_HIGH_LEVEL if low_code.startswith(k.lower()))
                        high_code = INFERRED_HIGH_LEVEL[key]
                        flag = "inferred_partial"
                        print(f"⚠️ Partial match: '{low_code}' → '{high_code}' in {ppt_id}, task {task_id}, thought {t.get('id')}")
                    else:
                        flag = "unmapped"
                        high_code = None
                        print(f"❌ Unmapped: '{low_code}' in {ppt_id}, task {task_id}, thought {t.get('id')}")

                all_data.append({
                    "ppt_id": ppt_id,
                    "task_id": task_id,
                    "thought_id": t.get("id", ""),
                    "text": text,
                    "low_level_code": low_code,
                    "high_level_code": high_code,
                    "token_count": count_tokens(text),
                    "flag": flag
                })
    return pd.DataFrame(all_data)

# === Load accuracy/reaction scores ===
def load_human_scores():
    df = pd.read_excel(HUMAN_SCORE_PATH)
    df["ppt_id"] = "p" + df["ppt_id"].astype(str)
    return df

# === Merge ===
def merge_human_data(thought_df, score_df):
    return thought_df.merge(score_df, on=["ppt_id", "task_id"], how="left")

# === Main function ===
def load_all_human_data(save_csv=True):
    print("📥 Loading thoughts...")
    df_thoughts = load_human_thought_data()
    print(f"✅ {len(df_thoughts)} thoughts loaded.")

    print("📥 Loading scores...")
    df_scores = load_human_scores()
    print(f"✅ {len(df_scores)} accuracy rows loaded.")

    df_merged = merge_human_data(df_thoughts, df_scores)
    print(f"🔗 Merged: {df_merged.shape[0]} rows")

    # Sort and reorder columns
    df_merged["ppt_id_num"] = df_merged["ppt_id"].str.extract(r"p(\d+)").astype(int)
    df_merged = df_merged.sort_values(by=["task_id", "ppt_id_num", "thought_id"])
    df_merged = df_merged.drop(columns=["ppt_id_num"])

    # Auto-assign OpenAI for unmapped only
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

    # Reorder columns
    column_order = [
        "task_id", "ppt_id", "thought_id", "text",
        "high_level_code", "low_level_code", "token_count",
        "flag", "accuracy", "reaction"
    ]
    df_merged = df_merged[column_order]

    if save_csv:
        df_merged.to_csv(OUTPUT_PATH, index=False)
        print(f"💾 Saved to {OUTPUT_PATH}")

    return df_merged

# === Run Script ===
if __name__ == "__main__":
    df_human = load_all_human_data()
    print(df_human.head())
