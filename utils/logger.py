from inspect import stack
import json
import pathlib
from datetime import datetime
from typing import Dict, Optional
from typing_extensions import TypedDict

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
LOG_FILE = RUNS_DIR / "log.jsonl"


class RunLog(TypedDict, total=False):
    run_id: int
    timestamp: str
    model: str
    configuration: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    scene_id: int
    agent_type: str
    rng_seed: int
    pmf_source: str
    achieved_steps: int
    step_count_violation: bool
    step_length_violation: bool
    scene_cap_violation: bool
    leakage: bool
    early_exit: bool
    target_met: bool


def next_run_id() -> int:
    """Return the next integer run identifier."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if not LOG_FILE.exists():
            return 1
        last_id = 0
        with LOG_FILE.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    data = json.loads(line)
                    last_id = max(last_id, int(data.get("run_id", 0)))
                except json.JSONDecodeError:
                    continue
        return last_id + 1
    except OSError as exc:
        raise RuntimeError("Unable to compute next run id") from exc


def log(entry: RunLog) -> None:
    """Append a log entry as JSONL."""
    required_keys = {
        "run_id",
        "timestamp",
        "model",
        "configuration",
        "prompt_tokens",
        "completion_tokens",
        "scene_id",
    }
    entry_keys = set(entry)
    if not required_keys.issubset(entry_keys):
        missing = required_keys - entry_keys
        raise ValueError(f"Log entry missing keys: {', '.join(sorted(missing))}")

    valid_keys = set(RunLog.__annotations__)
    unknown = entry_keys - valid_keys
    if unknown:
        raise ValueError(f"Unknown log keys: {', '.join(sorted(unknown))}")

    caller = stack()[1].frame.f_globals.get("name", "")
    agent_type = (
        "baseline"
        if "baselineLLM" in caller
        else "buffered" if "bufferedLLM" in caller else "other"
    )
    if "agent_type" not in entry:
        entry["agent_type"] = agent_type

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            json.dump(entry, fh)
            fh.write("\n")
    except OSError as exc:
        raise RuntimeError("Unable to write log entry") from exc


def write(entry: RunLog) -> None:
    """Append a raw log entry without modification."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            json.dump(entry, fh)
            fh.write("\n")
    except OSError as exc:
        raise RuntimeError("Unable to write log entry") from exc


# --------------------------------------------------------------------- #
# Public singleton instance for convenient imports
# --------------------------------------------------------------------- #
import types as _t

logger = _t.SimpleNamespace(
    next_run_id=next_run_id,
    log=log,
    write=write,
)

__all__ = ["RunLog", "next_run_id", "log", "write", "logger"]
