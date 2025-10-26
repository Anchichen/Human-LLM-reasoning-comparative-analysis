# baseline_agent.py
"""
Baseline Zendo Reasoning Agent
--------------------------------
Minimal, single‑pass LLM caller that emulates the human study’s procedure
without *any* cognitive scaffolds (no buffer, no reflection, no planning).

Workflow
~~~~~~~~
1. Loads a plain‑text prompt template (base_prompt.txt).
2. Reads a JSON file per task that contains the six scene descriptions.
3. Injects the scene text into the template.
4. Sends **one** completion request to the OpenAI API 
5. Saves the raw chain‑of‑thought response to an output folder.
6. Logs run metadata to `runs/log.jsonl` for later analysis.

The script intentionally avoids recursion, self‑checks, or extra vision calls
so it can serve as the reference condition for all subsequent interventions.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from utils import logger
from utils.config import PROJECT_ROOT, RUNS_DIR, TASK_DIR, get_openai_client
from utils.token_guard import estimate_token_count
from utils.scene_parser import panel_to_text
from utils.openai_adapter import create_completion

# Import the helper from buffered agent
from agents.bufferedLLM import add_target_steps

# Configuration (edit paths as needed)
PROMPT_PATH = PROJECT_ROOT / "prompts" / "base_prompt.txt"
EXAMPLE_PATH = PROJECT_ROOT / "prompts" / "chain_of_thought_example.txt"
OUTPUT_DIR = RUNS_DIR

MODEL_NAME = "gpt-4o"  # or other model with >=120 k context
TEMPERATURE = 0.3
MAX_TOKENS = 2048

# --- Load human-derived scene statistics once ---
SCENE_STATS = add_target_steps(
    json.loads(
        Path(PROJECT_ROOT / "configs" / "scene_stats.json").read_text(encoding="utf-8")
    )
)


def load_prompt(path: pathlib.Path) -> tuple[str, str]:
    """
    Return (template_text, placeholder) supporting both the *old*
    literal placeholder and the new {{SCENE_DESCRIPTION}} token.
    """
    text = path.read_text(encoding="utf-8")
    old_ph = "(Insert detailed text descriptions for Panels A–F here.)"
    new_ph = "{{SCENE_DESCRIPTION}}"

    if new_ph in text:
        return text, new_ph
    if old_ph in text:
        return text, old_ph
    raise ValueError(
        f"Placeholder missing from base prompt: expected {old_ph!r} or {new_ph!r}"
    )



def build_scene_text(scene_file: pathlib.Path) -> str:
    """Convert a task JSON into the textual panel descriptions expected by the prompt.
    """
    with scene_file.open("r", encoding="utf-8") as fh:
        data: Dict = json.load(fh)
        
    panels: Dict[str, Dict] = data["panels"]
    ordered_text: List[str] = [
        f"Panel {label}: {panel_to_text(panels[label])}"
        for label in sorted(panels)
    ]
    return "\n".join(ordered_text)


def render_prompt(template: str, placeholder: str, scene_text: str) -> str:
    """Inject scene description at whichever placeholder the template uses."""
    return template.replace(placeholder, scene_text)



def call_llm(prompt: str, model_name: str, *, max_tokens: int | None = None) -> Tuple[str, object | None]:
    """Single deterministic completion call using the OpenAI SDK v1."""
    import openai

    # Use the provided max_tokens or fall back to the default
    max_tokens = max_tokens if max_tokens is not None else MAX_TOKENS

    response = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content.strip()
    return content, getattr(response, "usage", None)


def run_task(
    task_id: int,
    *,
    outdir: pathlib.Path = OUTPUT_DIR,
    force: bool = False,
    dry_run: bool = False,
    model_name: str = MODEL_NAME,
    batch_run_id: int | None = None
) -> pathlib.Path:
    from utils.config import get_openai_client

    client = get_openai_client(dry_run)

    # prompt of the Zendo task
    prompt_template, placeholder = load_prompt(PROMPT_PATH)
    if placeholder not in prompt_template:
        raise ValueError("Placeholder missing from base prompt")
    
    # example from human chain of thought; not provided so far
    example_text = Path(EXAMPLE_PATH).read_text(encoding="utf-8")

    # scene description
    task_file = TASK_DIR / f"Task{task_id}.json"
    if not task_file.exists():
        raise FileNotFoundError(f"Scene file not found: {task_file}")
    scene_text = build_scene_text(task_file)


    # Combine: base system instruction + scene description
    prompt_text = render_prompt(prompt_template, placeholder, scene_text)       # Human example if needed: (example_text + "\n\nNow it’s your turn. Follow the same step-by-step style below.\n\n")

    full_prompt = prompt_text
    prompt_tokens = estimate_token_count(full_prompt) # tokens of the full prompt
    
    # --- Compute human-based constraints for this scene ---
    stats = SCENE_STATS[f"scene{task_id}"]
    
    min_steps = int(stats["median_thoughts"] - stats["iqr_thoughts"]/2)
    max_steps = int(stats["median_thoughts"] + stats["iqr_thoughts"]/2)
    low_tok   = int(stats["median_tokens"] - stats["iqr_tokens"]/2)
    high_tok  = int(stats["median_tokens"] + stats["iqr_tokens"]*2)       # set the upper bound of thought token higher to allow more flexible thinking. 
    scene_cap = int(stats["median_scene_tokens"] + stats["iqr_scene_tokens"]/2)

    # print(f"[DEBUG] min_steps: {min_steps}, max_steps: {max_steps}, low_tok: {low_tok}, high_tok: {high_tok}")  # print the stats to check

    if not force and prompt_tokens + MAX_TOKENS > 8_000:
        print(
            f"Skipping Task {task_id}: {prompt_tokens + MAX_TOKENS} tokens exceed 8000"
        )
        return outdir / f"skipped_task{task_id}.txt"
    
    # ─── Prompt‐based self‐regulation instruction of steps and tokens───
    constraint_instructions = (
        f"\n\nPlease produce between {min_steps} and {max_steps} steps, "
        f"each thought should be at least {low_tok} tokens."
    )
    full_prompt += constraint_instructions

    # BEFORE the dry_run check, initialize post-hoc vars to safe defaults:
    segments = []
    actual_steps = 0
    step_token_lengths = []
    step_length_violations = []
    scene_cap_violation = False

    # content showed for dry-run
    lorem = "lorem ipsum dolor sit amet consectetur " "adipiscing elit sed do eiusmod"

    if dry_run:
        response_text = lorem
        usage_prompt = usage_completion = 0

    # real run, calling openai agent
    else:
        response_text, usage = call_llm(
            full_prompt,
            model_name,
            max_tokens = MAX_TOKENS
            )
        usage_prompt = usage.prompt_tokens if usage else 0
        usage_completion = usage.completion_tokens if usage else 0

        # --- Post‐hoc validation of steps & tokens ---
        segments = re.split(r"\[T\d{2}\]", response_text.strip())
        actual_steps = len([s for s in segments if s.strip()])  # only count non-empty segments

        step_token_lengths = []
        step_length_violations = []
        for seg in segments:
            if seg.strip():
                tok_count = estimate_token_count(seg.strip())
                step_token_lengths.append(tok_count)
                step_length_violations.append(not (low_tok <= tok_count <= high_tok))

        # scene‐cap violation on response
        scene_cap_violation = estimate_token_count(response_text) > scene_cap



    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    # Use the shared batch_run_id if provided, else allocate a new one
    if batch_run_id is not None:
        run_id = batch_run_id
    else:
        run_id = logger.next_run_id()

     # ─── Create a per-run folder ───
    run_folder = outdir / f"run{run_id:02d}"
    run_folder.mkdir(parents=True, exist_ok=True)

    
    timestamp = datetime.utcnow().isoformat()
    timestamp_str = datetime.now().strftime("[%Y-%m-%d %H:%M]   ")
    header = f"[run{run_id:02d}_task{task_id}]   {timestamp_str}\n\n"
    response_text = header + response_text.strip()


    out_file = run_folder / f"run{run_id:02d}_task{task_id}.txt"
    out_file.write_text(response_text, encoding="utf-8")

    # metadata record in log.jsonl
    log_entry = {
        "run_id": run_id,
        "scene_id": task_id,
        "model": model_name,
        "configuration": "baseline",
        "timestamp": timestamp,
        "prompt_tokens": usage_prompt if not dry_run else 0,
        "completion_tokens": usage_completion if not dry_run else 0,
        "output_file": str(
            out_file.relative_to(PROJECT_ROOT)
            if PROJECT_ROOT in out_file.parents
            else out_file
        ),
        "step_count_violation": not (min_steps <= actual_steps <= max_steps),
        "actual_steps": actual_steps,
        "step_length_violation": any(step_length_violations),
        "step_token_lengths": step_token_lengths,
        "scene_cap_violation": scene_cap_violation,
    }
    logger.write(log_entry)

    # ─── Append Final Guesses ───
    # build final-guess prompt with full scene + full chain-of-thought
    user_msg = {
        "role": "user",
        "content": (
            "You have finished reasoning.\n\n"
            "Based on the reasoning steps above, list up to three specific and concise possible rules that explain when the star appears in order of confidence.\n"
            "Each rule should describe features like cone color, size, orientation, or their combinations."
            "Be precise — e.g., 'The rule is there must be at least one large green cone.', 'The star appears only when two cones are stacked with matching colors.', 'There are exactly three red small cones and they are all tilted when the panels follow the rule.'"
            "Label each guess [G01], [G02], [G03]. Do not repeat prior observations — just claim your final guesses of the rule."
            )
        }
    # build messages array for final guesses
    messages = [{"role":"system","content":render_prompt(prompt_template, placeholder, scene_text)}]
    for idx, seg in enumerate(segments[1:], start=1):
        messages.append({"role":"assistant","content":f"[T{idx:02d}] {seg.strip()}"})
    messages.append(user_msg)

    # get final guesses
    summary_resp = create_completion(client, model_name, messages=messages, temperature=0.3)
    final_guesses = summary_resp.choices[0].message.content.strip()

    # Append cleaned final guesses (one per line) to the output .txt file
    # Ensure each guess starts on a new line
    with open(out_file, "a", encoding="utf-8") as f:
        f.write("\n\n\n---Final Guesses---\n")
        for line in final_guesses.splitlines():
            if line.strip().startswith("[G"):
                f.write(line.strip() + "\n")

    guesses_cleaned = "\n".join(line.strip() for line in final_guesses.splitlines() if line.strip().startswith("[G"))
    # also save to a separate JSONL for guesses
    guess_log = {
        "run_id": run_id,
        "task_id": task_id,
        "final_guesses": [line.strip() for line in guesses_cleaned.splitlines() if line.strip()],
        "accuracy": None
    }

    with open(PROJECT_ROOT / "runs" / "baseline_final_guesses.jsonl", "a", encoding="utf-8") as gf:
        gf.write(json.dumps(guess_log) + "\n")



    # ─── Append to master task and run files ───
    task_master_file = PROJECT_ROOT / "runs" / f"task{task_id}_all.txt"
    run_master_file = run_folder / f"run{run_id:02d}_all.txt"

    # Content block for each run
    full_block = response_text + "\n\n---Final Guesses---\n" + final_guesses.strip() + "\n\n"

    # Append to per-task file
    with open(task_master_file, "a", encoding="utf-8") as tf:
        tf.write(full_block + "\n")

    # Append to per-run file (in run folder)
    with open(run_master_file, "a", encoding="utf-8") as rf:
        rf.write(full_block + "\n")


    # return the final output file
    return out_file


# Main execution

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Return parsed CLI arguments.
    argv:
        Optional list of command line arguments. When ``None`` (default),
        ``argparse`` parses ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description="Baseline Zendo Reasoning Agent")
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Task IDs to run (e.g. --tasks 1 2 3)",
    )
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=OUTPUT_DIR,
        help="Directory for raw outputs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore token guard and force API call",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run locally without real OpenAI calls.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    for task_idx in args.tasks:
        out_file = run_task(
            task_idx,
            outdir=args.outdir,
            force=args.force,
            dry_run=args.dry_run,
        )
        print(f"✓ Task {task_idx} complete → {out_file.name}")

    print(f"Baseline run finished. Logs stored in {logger.RUNS_DIR}.")


if __name__ == "__main__":
    main()
