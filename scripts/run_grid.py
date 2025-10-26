from __future__ import annotations

import argparse
import json
import pathlib
from itertools import product
from typing import Dict, Iterable, Tuple

from agents import bufferedLLM

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
LOG_FILE = RUNS_DIR / "log.jsonl"


Key = Tuple[int, int, str]


def load_existing(log_path: pathlib.Path) -> set[Key]:
    existing: set[Key] = set()
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (
                    data.get("task_id"),
                    data.get("buffer_size"),
                    data.get("model_name"),
                )
                if None not in key:
                    existing.add(key)  # type: ignore[arg-type]
    return existing


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bufferedLLM grid")
    parser.add_argument("--buffers", nargs="+", type=int, required=True)
    parser.add_argument("--models", nargs="+", type=str, required=True)
    parser.add_argument("--tasks", nargs="+", type=int, required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def print_progress(status: Dict[Key, str]) -> None:
    header = f"{'task':^4} | {'buf':^4} | {'model':^15} | {'status':^7}"
    print(header)
    print("-" * len(header))
    for (task_id, buffer_size, model_name), st in sorted(status.items()):
        model_disp = (model_name[:12] + "...") if len(model_name) > 15 else model_name
        print(f"{task_id:^4} | {buffer_size:^4} | {model_disp:^15} | {st:^7}")
    print()


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    combos = list(product(args.tasks, args.buffers, args.models))
    existing = load_existing(LOG_FILE)
    status: Dict[Key, str] = {combo: "pending" for combo in combos}
    for combo in combos:
        if combo in existing:
            status[combo] = "skip"
            print_progress(status)
            continue
        task_id, buffer_size, model = combo
        bufferedLLM.run_task(task_id, buffer_size, model)
        status[combo] = "done"
        print_progress(status)


if __name__ == "__main__":
    main()
