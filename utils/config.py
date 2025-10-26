from __future__ import annotations

from pathlib import Path
import os
from importlib.metadata import PackageNotFoundError, version
import types
import warnings

# Paths used across the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = PROJECT_ROOT / "Task"
RUNS_DIR = PROJECT_ROOT / "runs"

# Ensure the runs directory exists
RUNS_DIR.mkdir(exist_ok=True, parents=True)

# Environment / API key handling
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ModuleNotFoundError:  # python-dotenv not installed
    pass


def require_api_key() -> str:
    """Return the OpenAI API key or raise if missing or invalid."""

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found. Set env var or use --dry-run.")
    if any(ord(ch) > 127 for ch in key):
        raise RuntimeError("OPENAI_API_KEY contains non-ASCII characters")
    return key


def get_openai_client(dry_run: bool):
    """Return an object with ``.chat.completions.create``."""

    import openai

    if dry_run:
        class _Stub:
            class chat:
                class completions:
                    @staticmethod
                    def create(**_kw):
                        return types.SimpleNamespace(
                            usage={"prompt_tokens": 0, "completion_tokens": 0},
                            choices=[
                                types.SimpleNamespace(
                                    message=types.SimpleNamespace(
                                        content="<stub>"
                                    )
                                )
                            ],
                        )

        return _Stub()

    key = require_api_key()
    return openai.OpenAI(api_key=key)


def set_openai_key(dry_run: bool) -> None:
    warnings.warn(
        "set_openai_key is deprecated; use get_openai_client", DeprecationWarning
    )
    globals()["_CLIENT"] = get_openai_client(dry_run)
