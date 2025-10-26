# bufferedLLM.py
from __future__ import annotations
from utils.openai_adapter import create_completion
from tqdm import trange

import argparse
import json
import openai
import tiktoken
import pathlib
import re
from pathlib import Path
import random
import time
from typing import List

import sys
import types
from utils.config import get_openai_client
import os
from datetime import datetime
from utils.scene_parser import panel_to_text


def add_target_steps(stats: dict) -> dict:
    """Return a copy with ``target_steps`` computed when applicable."""

    if not isinstance(stats, dict):
        return stats

    new_stats = {
        k: add_target_steps(v) if isinstance(v, dict) else v for k, v in stats.items()
    }

    if {"median_thoughts", "iqr_thoughts"} <= stats.keys():
        new_stats["target_steps"] = int(
            stats["median_thoughts"] + stats["iqr_thoughts"] / 2
        )

    if {"median_tokens", "iqr_tokens"} <= stats.keys():
        if "lower_tok" not in new_stats:
            new_stats["lower_tok"] = stats["median_tokens"] - stats["iqr_tokens"] / 2
        if "upper_tok" not in new_stats:
            new_stats["upper_tok"] = stats["median_tokens"] + stats["iqr_tokens"] / 2

    return new_stats



__all__: list[str] = []

enc = None
from utils import logger
from utils.config import PROJECT_ROOT, RUNS_DIR, TASK_DIR
from utils.token_guard import (
    estimate_token_count,
    enforce_buffer,
    max_context as MAX_CONTEXT,
    TokenLimitError,
)

# Re-export for test fixtures
from utils.config import TASK_DIR, RUNS_DIR, PROJECT_ROOT

__all__ += ["TASK_DIR", "RUNS_DIR", "PROJECT_ROOT"]

PROMPT_PATH = PROJECT_ROOT / "prompts" / "base_prompt.txt"

SCENE_STATS = add_target_steps(json.loads(Path("configs/scene_stats.json").read_text()))

# regular-expression helpers
_TAG_RE = re.compile(r"^\[T\d{2}\]\s*")          
_NUM_RE = re.compile(r"^\d+[\.\)]\s*")     
# remove a leading tag and leading list number from ONE line
def _clean(line: str) -> str:
    return _NUM_RE.sub("", _TAG_RE.sub("", line)).strip()

class BufferedLLM:
    def __init__(
        self,
        buffer_size: int = 3,
        model_name: str | None = None,
        seed: int | None = None,
        dry_run: bool | None = None,
    ) -> None:
        self.rng_seed = seed
        self.buffer_size = buffer_size
        self.model_name = model_name or "gpt-3.5-turbo"
        self.dry_run = dry_run
        self.client = get_openai_client(bool(dry_run))
        template = Path(PROMPT_PATH).read_text(encoding="utf-8")
        self.initial_system_message = template
        self.early_exit = False
        self.thoughts: list[dict] = []
        self.step_length_violation = False
        self.step_count_violation = False
        self.scene_cap_violation = False
        self.target_met = False
        self.achieved_steps = 0
        self.step_token_lengths: list[int] = []
        self.step_length_flags: list[bool] = []
        if seed is not None:
            random.seed(seed)

        # --- buffer-fix START --
        # Monotonic counter for tag prefix [TXX]
        self.global_step = 0
        # --- buffer-fix END --

        self.stats_path = (
            Path(__file__).resolve().parents[1] / "configs" / "scene_stats.json"
        )
        with self.stats_path.open(encoding="utf-8") as fp:
            self.scene_stats = add_target_steps(json.load(fp))

    def run_task(
        self,
        task_id: int,
        buffer_size: int | None = None,
        model_name: str | None = None,
        *,
        dry_run: bool | None = None,
    ) -> pathlib.Path:
        if dry_run is None:
            dry_run = self.dry_run
        self.dry_run = dry_run
        self.client = get_openai_client(bool(dry_run))

        global enc
        if enc is None:
            try:
                import tiktoken

            except ModuleNotFoundError:
                tiktoken = types.ModuleType("tiktoken")
                sys.modules["tiktoken"] = tiktoken

                class _StubEncoding:
                    def encode(self, text: str, **_kw) -> list[int]:
                        return list(range(len(text.split())))

                tiktoken.encoding_for_model = lambda _model: _StubEncoding()
                tiktoken.get_encoding = lambda _name: _StubEncoding()

            enc = tiktoken.get_encoding("cl100k_base")

        if buffer_size is None:
            buffer_size = self.buffer_size
        else:
            self.buffer_size = buffer_size
        if model_name is None:
            model_name = self.model_name
        return _run_task(task_id, buffer_size, model_name, self.rng_seed, self)     # calls the full logic function with agent

    def _build_messages(self, scene_desc: str) -> list[dict]:
        """Return chat messages constructed from the scene and recent thoughts."""

        # keep only last N assistant thoughts *without tags or list numbers*
        history = [
            {"role": "assistant", "content": _clean(t["content"])}
            for t in self.thoughts[-self.buffer_size :]
        ]
        return [{"role": "system", "content": scene_desc}] + history


def _build_scene_text(task_file: pathlib.Path) -> str:
    data = json.load(task_file.open())
    panels: dict[str, dict] = data["panels"]
    lines = []
    for label in sorted(panels):
        lines.append(f"Panel {label}: {panel_to_text(panels[label])}")
    return "\n".join(lines)

# the core logic function
def _run_task(
    task_id: int,
    buffer_size: int,
    model_name: str,
    rng_seed: int | None,
    agent: BufferedLLM | None = None,
) -> pathlib.Path:
    if buffer_size < 1:
        raise ValueError("buffer_size must be >= 1")

    if agent is not None:
        agent.early_exit = False
        agent.thoughts.clear()
        agent.buffer_size = buffer_size

    task_file = TASK_DIR / f"Task{task_id}.json"
    template = Path(PROMPT_PATH).read_text()
    example = Path(PROJECT_ROOT / "prompts" / "chain_of_thought_example.txt").read_text()
    scene_text = _build_scene_text(task_file)
    # First show the human example, then the actual task
    prompt_text = (
       example
       + "\n\nNow it’s your turn. Follow the same step-by-step style below.\n\n"
       + template.replace("{{SCENE_DESCRIPTION}}", scene_text)
   )
    stats = SCENE_STATS[f"scene{task_id}"]
    med_thoughts = stats["median_thoughts"]
    band_thoughts = (
        med_thoughts - stats["iqr_thoughts"] / 2,
        med_thoughts + stats["iqr_thoughts"] / 2,
    )
    target_steps = random.randint(
        int(band_thoughts[0]),
        int(band_thoughts[1])
    )

    med_tokens = stats["median_tokens"]
    scene_cap = stats["median_scene_tokens"] + (stats["iqr_scene_tokens"] / 2)
    token_band = (
        med_tokens - stats["iqr_tokens"] / 2,
        med_tokens + stats["iqr_tokens"] / 2,
    )
    lower_tok, upper_tok = token_band

    scene_desc = (
        f"{prompt_text}\n"
        f"Scene {task_id}\n"
        f"\u2022 Produce exactly one novel reasoning step per API call. The last N assistant messages shown are your own prior thoughts; do not restate the same ideas from prior thoughts (T)\n"
        f"\u2022 Each new step must introduce a _distinct_ observation. Do not reuse, paraphrase, or restate any idea you’ve previously given in this chain of thought.\n"
        f"\u2022 On the **{target_steps:02d}**th call, at the end of your step include exactly `### FINAL RULE:` followed by the final rule.\n"
        f"\u2022 Keep each step ~{med_tokens} tokens ({token_band[0]:.0f}–{token_band[1]:.0f} is fine).\n"
        f"\u2022 Total words for the scene must not exceed {scene_cap:.0f}.\n"
    )

    static_prompt = scene_desc

    retry_count = 0
    early_exit = False
    step_token_lengths: list[int] = []
    step_length_flags: list[bool] = []
    scene_cap_violation = False

    scene_tokens = estimate_token_count(scene_desc)

    lorem = "lorem ipsum dolor sit amet consectetur " "adipiscing elit sed do eiusmod"

    # loop exactly target_steps times
    for step_num in trange(1, target_steps + 1, desc=f"Task {task_id}", unit="step"):
        # --- build a per-step prompt that tells GPT exactly which step we're on
        dynamic_prefix = (
            f"Step {step_num} of {target_steps}.\n"
            f"Do NOT repeat any content from Steps 1–{step_num-1}.\n"
        )
        # assemble final scene_desc this round
        round_prompt = dynamic_prefix + static_prompt

        # construct messages with this per-step prompt
        messages = agent._build_messages(round_prompt)
        usage_prompt = usage_completion = 0
        history = agent.thoughts[-buffer_size:]
        buffer_tokens = (
            estimate_token_count("".join(m["content"] for m in history)) + 600
        )
        enforce_buffer(scene_tokens, buffer_tokens, MAX_CONTEXT)
        if len(enc.encode(scene_desc + "".join(m["content"] for m in history))) > 4000:
            scene_cap_violation = True
            agent.thoughts.append(
                {"role": "assistant", "content": "### ABORT SCENE CAP"}
            )
            break

        # if doing dry run, show fake outputs
        if agent and agent.dry_run:
            if step_num == target_steps:
                reply_core = "### FINAL RULE: stub rule"
            else:
                reply_core = lorem
            reply = reply_core
            usage_prompt = usage_completion = 0
        
        # calling api to do actual run
        else:
            try:
                response = create_completion(
                    agent.client,
                    model_name,
                    messages=messages,
                    temperature=0.7,
                )
            # retry after waiting; if retry over 5 times, break
            except openai.RateLimitError:
                if retry_count == 5:
                    break
                time.sleep(2**retry_count)
                retry_count += 1
                continue

            # clean the raw output, remove tag and number 
            raw = response.choices[0].message.content
            # strip tag at very start, strip any remaining tags, strip leading numbers
            reply = _TAG_RE.sub("", raw)
            reply = re.sub(r"\[T\d{2}\]\s*", "", reply)
            reply = _NUM_RE.sub("", reply) 
            
            # duplicate thought detection
            from difflib       import SequenceMatcher   # compare string similarity
            from nltk.stem     import PorterStemmer     # simplify words to their root
            ps = PorterStemmer()

            SIM_RATIO  = 0.75   # similarity by character (surface text)
            JACC_RATIO = 0.65   # similarity by word meaning
            MAX_RETRIES = 3

            # clean, lower, and stemming the text
            def _normalize(txt: str) -> set[str]:
                words = re.findall(r"[A-Za-z]+", txt.lower())
                return {ps.stem(w) for w in words}

            # main check of similarity 
            def _too_similar(a: str, b: str) -> bool:
                if SequenceMatcher(None, a, b).ratio() >= SIM_RATIO: 
                    return True                                         # similarity by character
                
                wa, wb = _normalize(a), _normalize(b)
                if not wa or not wb:
                    return False
                j = len(wa & wb) / len(wa | wb)                         # similarity by word meaning
                return j >= JACC_RATIO
            
            dup_retries = 0
            while True:
                # … after you get `reply` from the model … clean the reply and previous thoughts
                clean_reply = _clean(reply)
                past_clean  = [_clean(m["content"]) for m in agent.thoughts]

                if any(_too_similar(clean_reply, prev) for prev in past_clean):
                    dup_retries += 1
                    if dup_retries >= MAX_RETRIES:
                        # give up and accept it once (to avoid infinite loops)
                        break
                    # ask for a *different* idea
                    messages.append({
                        "role": "user",
                        "content": (
                            "That idea is too similar to one you’ve already stated."
                            "Provide a new, non-repetitive reasoning step describing a different aspect of the panels."
                        ),
                    })
                    continue  # re-call the model, don’t advance step_num yet (go back to the top of the loop and try again with the updated message list)
                else:
                    # unique enough—reset counter and accept
                    dup_retries = 0
                    break

            # CONTRACT CHECK START 
            premature_final = (step_num < target_steps and "### FINAL RULE" in reply) # if generate final rule before on the last step

            # exit early if retry over 5 times
            if premature_final:
                retry_count += 1
                if retry_count > 5:
                    early_exit = True
                    break
                continue
            # CONTRACT CHECK END 

            # taken usage logging
            retry_count = 0
            usage_prompt = (
                response.usage.prompt_tokens
                if getattr(response, "usage", None) is not None
                else 0                                              # dry-run condition
            )
            usage_completion = (
                response.usage.completion_tokens
                if getattr(response, "usage", None) is not None
                else 0                                              # dry-run condition
            )


        # --- TOKEN-LENGTH ENFORCEMENT START ---
            # Automatically adjust any step that falls outside [lower_tok, upper_tok].
            if os.getenv("ALLOW_LONG_STEPS") != "1":
                toklen = len(enc.encode(reply))
                adjust_attempts = 0
                while (toklen < lower_tok or toklen > upper_tok) and adjust_attempts < 3:
                    # Ask the model to restate the step within the allowed length
                    adjust_prompt = {
                        "role": "user",
                        "content": (
                            f"Your previous step was {toklen} tokens long. "
                            f"Please restate it using between {lower_tok} and {upper_tok} tokens."
                        ),
                    }
                    messages.append(adjust_prompt)
                    adj_resp = create_completion(           # get new reply
                        agent.client,
                        model_name,
                        messages=messages,
                        temperature=0.7,
                    )
                    reply = adj_resp.choices[0].message.content.strip() 
                    toklen = len(enc.encode(reply))         # calculate new token length
                    adjust_attempts = 1                     # only allow one retry, accept whatever after one retry
                step_length_violation = False
            # --- TOKEN-LENGTH ENFORCEMENT END ----

        # we know exactly which step this is
        tag = f"[T{step_num:02d}] "
        tag_tokens = len(enc.encode(tag))
        if agent and agent.dry_run:
            eff_len = len(enc.encode(reply))
        else:
            eff_len = len(enc.encode(reply)) - tag_tokens

        lower, upper = token_band
        if agent and agent.dry_run and "### FINAL RULE" in reply:
            too_short = too_long = False

        else:
            too_short = eff_len < lower
            too_long = eff_len > upper
        step_length_flags.append(too_short or too_long)
        step_token_lengths.append(eff_len)
        full_thought = tag + reply
        agent.thoughts.append({"role": "assistant", "content": full_thought})
        if step_num == target_steps and "### FINAL RULE" not in reply:
            reply += "\n### FINAL RULE: <YOUR HYPOTHESIS HERE>"                 # add a placeholder
        if too_short or too_long:
            if retry_count == 5:
                break
            retry_count += 1
            continue
        retry_count = 0

    # Count how many tokens the final prompt used
    total_prompt_tokens = len(
        enc.encode(
            scene_desc + "".join(m["content"] for m in agent.thoughts[-buffer_size:])
        )
    )

    achieved_steps = len(agent.thoughts)
    step_count_violation = achieved_steps != target_steps
    target_met = (achieved_steps == target_steps) and not early_exit
    step_length_violation = any(step_length_flags)
    scene_cap_violation = scene_cap_violation or total_prompt_tokens > 4000     # whole prompt token

    # check if it uses past thoughts that do not exist in memory buffer
    chain_text = "\n".join(t["content"] for t in agent.thoughts)
    min_allowed = agent.global_step - buffer_size + 1
    leakage = any(
        int(m.group(1)) < min_allowed for m in re.finditer(r"\[T(\d{2})\]", chain_text)
    )

    # output file 
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RUNS_DIR / f"task{task_id}_buffer{buffer_size}.txt"

    # add time stamp
    stamp = datetime.utcnow().strftime("%d%m%Y_%H%M")
    with open(out_file, "a", encoding="utf-8") as f:    
        f.write(stamp + "\n")
        f.write("\n".join(t["content"] for t in agent.thoughts))
        f.write("\n\n")  

    # metadata record logging, once per task run
    run_id = logger.next_run_id()
    log_entry = {
        "run_id": run_id,
        "task_id": task_id,
        "buffer_size": buffer_size,
        "model_name": model_name,
        "prompt_tokens": usage_prompt if not agent.dry_run else 0,
        "completion_tokens": usage_completion if not agent.dry_run else 0,
        "output_file": str(out_file.relative_to(PROJECT_ROOT)),
        "step_length_violation": step_length_violation,
        "step_count_violation": step_count_violation,
        "scene_cap_violation": scene_cap_violation,
        "rng_seed": rng_seed,
        "pmf_source": "empirical_steps_v1",
        "step_token_lengths": step_token_lengths,
        "step_length_flags": step_length_flags,
        "target_steps": target_steps,
        "achieved_steps": achieved_steps,
        "early_exit": early_exit,
        "target_met": target_met,
        "leakage": leakage,
    }
    logger.write(log_entry)

    if agent is not None:
        agent.step_length_violation = step_length_violation
        agent.step_count_violation = step_count_violation
        agent.scene_cap_violation = scene_cap_violation
        agent.target_met = target_met
        agent.achieved_steps = achieved_steps
        agent.early_exit = early_exit
        agent.step_token_lengths = step_token_lengths
        agent.step_length_flags = step_length_flags

    # ─── Final‐guess step: use only the last `buffer_size` thoughts ───
    # after the whole chain-of-thought, give up to three final rule guesses
    system_msg = static_prompt

    # Collect the last `buffer_size` assistant messages
    buffer_msgs = agent.thoughts[-buffer_size:]

    # Build the final‐guess prompt
    final_messages = []
    final_messages.append({"role": "system",  "content": system_msg})
    for th in buffer_msgs:
        final_messages.append({"role": th["role"], "content": th["content"]})
    final_messages.append({
        "role": "user",
        "content": (
            "Based only on the reasoning steps above, "
            "list up to three concise candidate rules that could explain the hidden pattern. "
            "Number them 1., 2., 3. Do not include any of your prior observations or chain‐of‐thought—just the final rule hypotheses."
        )
    })

    # Call the model one more time
    summary_resp = create_completion(
        agent.client,
        model_name,
        messages=final_messages,
        temperature=0.0,
    )
    final_guesses = summary_resp.choices[0].message.content

    # Append those guesses to the same output file
    with open(out_file, "a") as f:
        f.write("\n\n# Final Rule Hypotheses\n")
        f.write(final_guesses + "\n\n")

    # Return the completed file

    return out_file

def run_task(
    task_id: int,
    buffer_size: int,
    model_name: str,
    *,
    dry_run: bool = False,
) -> pathlib.Path:
    agent = BufferedLLM(dry_run=dry_run)                                        # creates an agent instance
    return agent.run_task(task_id, buffer_size, model_name, dry_run=dry_run)    # calls the method in the class


# Legacy CLI shim so `python main.py buffered ...` continues to work

import argparse, json, sys
from utils.config import PROJECT_ROOT


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Buffered Zendo Agent (shim)")
    p.add_argument("--tasks", nargs="+", type=int, default=[1])
    p.add_argument(
        "--buffer",
        type=int,
        required=True,
        help="working-memory size (thoughts remembered)",
    )
    p.add_argument("--model", default="gpt-3.5-turbo")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_known_args(argv)[0]


def _load_scene_stats() -> dict:
    stats_path = PROJECT_ROOT / "configs" / "scene_stats.json"
    with stats_path.open(encoding="utf-8") as fh:
        return add_target_steps(json.load(fh))  

def main() -> None:  
    args = _parse_args()
    stats = _load_scene_stats()

    agent = BufferedLLM(
        buffer_size=args.buffer,
        model_name=args.model,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    for task_id in args.tasks:
        key = f"scene{task_id}" if f"scene{task_id}" in stats else str(task_id)
        if key not in stats:
            print(f"⚠️  Stats missing for Task {task_id} – skipped")
            continue
        s = stats[key]
        out_file = agent.run_task(
            task_id=task_id,
            scene_desc="dummy scene text (shim)",  # not used in dry-run
            target_steps=s["target_steps"],
            lower_tok=s["lower_tok"],
            upper_tok=s["upper_tok"],
        )
        print(f"✓ Task {task_id} complete → {out_file.name}")

    print("Buffered run finished. Logs stored in runs/")


# Allow `python agents/bufferedLLM.py --help`
if __name__ == "__main__":
    if {"-h", "--help"} & set(sys.argv):
        _parse_args()  # prints help & exits
        sys.exit(0)
    main()
