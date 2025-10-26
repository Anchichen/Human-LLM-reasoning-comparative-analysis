from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

from utils.config import TASK_DIR

_CONE_FIELDS = (
    "index",
    "size",
    "colour",
    "orientation",
    "contacts",
    "position",
)


def panel_to_text(panel: Dict | str) -> str:
    """
    Return one concise English paragraph for a panel.
    • Otherwise we synthesise a description from the cone table.
    """

    if isinstance(panel, str):
        return panel.strip()

    if "text" in panel:
        return str(panel["text"]).strip()

    cones: List[List] = panel["cones"]
    lines = []
    has_star = panel["yellow_star"]  # assumes key always exists

    # Make star presence extremely clear
    header = (
        "This panel HAS a yellow star. It follows the hidden rule."
        if has_star else
        "This panel DOES NOT have a yellow star. It does not follow the rule."
    )

    lines = [header]

    for row in cones:
        cone = dict(zip(_CONE_FIELDS, row))

        # Standardized format: Cone X: Size Color, Orientation — Contact
        desc = (
            f"- Cone {cone['index']}: {cone['size']} {cone['colour']}, "
            f"{cone['orientation']} — {cone['contacts']}."
        )

        lines.append(desc)

    # Each cone on its own line; easier for the LLM to parse.
    return "\n".join(lines)


def load_scene(task_id: int, *, task_dir: Path | None = None) -> Dict:
    """Return the parsed JSON for ``task_id`` from ``Task/TaskX.json``."""

    root = task_dir or TASK_DIR
    path = root / f"Task{task_id}.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
