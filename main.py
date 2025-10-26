"""Command-line entry-point for running baseline or buffered agents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from agents import baselineLLM, bufferedLLM
from utils import logger


# CLI PARSER 

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM Design Config CLI")
    sub = p.add_subparsers(dest="mode", required=True)

    # baseline
    b0 = sub.add_parser("baseline")
    b0.add_argument("--tasks", nargs="+", type=int, required=True)
    b0.add_argument("--outdir", type=Path, default=logger.RUNS_DIR / "baseline")
    b0.add_argument("--model", default=baselineLLM.MODEL_NAME)
    b0.add_argument("--seed", type=int, default=None)
    b0.add_argument("--dry-run", action="store_true")
    b0.add_argument("--runs", type=int, default=1)

    # buffered
    b1 = sub.add_parser("buffered")
    b1.add_argument("--tasks", nargs="+", type=int, required=True)
    b1.add_argument("--buffer", type=int, required=True)
    b1.add_argument("--model", default="gpt-3.5-turbo")
    b1.add_argument("--seed", type=int, default=None)
    b1.add_argument("--dry-run", action="store_true")
    b1.add_argument("--runs", type=int, default=1)

    return p


# MAIN LAUNCHER

def launch_agent(args: argparse.Namespace) -> None:
    if args.mode == "baseline":
            # Each full run (across all tasks) gets exactly one run_id
            for _ in range(args.runs):
                batch_run_id = logger.next_run_id()
                for tid in args.tasks:
                    out_file = baselineLLM.run_task(
                        task_id=tid,
                        outdir=args.outdir,
                        dry_run=args.dry_run,
                        model_name=args.model,
                        batch_run_id=batch_run_id,
                    )
                    print(f"✓ Run {batch_run_id} Task {tid} complete → {out_file.name}")

    elif args.mode == "buffered":
        agent = bufferedLLM.BufferedLLM(
            buffer_size=args.buffer,
            model_name=args.model,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        for tid in args.tasks:
            for _ in range(args.runs):
                out_file = agent.run_task(
                    tid,            # task_id  (positional)
                    args.buffer,    # buffer_size
                    args.model,     # model_name
                    dry_run=args.dry_run,
                )
                print(f"✓ Task {tid} complete → {out_file.name}")

    print(f"{args.mode.capitalize()} run finished. Logs stored in {logger.RUNS_DIR}")


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    launch_agent(args)


if __name__ == "__main__":
    main()
