from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import random

SentenceTransformer = None
cosine_similarity = None


def topic_centered_cosine(
    log_path: str,
    embed_model: str = "all-mpnet-base-v2",
    thought_key: str = "text",
    seed: int = 42,
) -> pd.DataFrame:
    """Return topic-centred cosine similarity metrics for a run log."""

    global SentenceTransformer
    global cosine_similarity
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST

        SentenceTransformer = ST
    if cosine_similarity is None:
        from sklearn.metrics.pairwise import cosine_similarity as CS

        cosine_similarity = CS

    path = Path(log_path)
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not records:
        raise ValueError("Log is empty")

    prompt: str | None = None
    thoughts: List[str] = []
    buffer_size = None

    for rec in records:
        if buffer_size is None and "buffer_size" in rec:
            buffer_size = rec["buffer_size"]
        role = rec.get("role")
        text = rec.get(thought_key)
        if role == "user" and prompt is None:
            prompt = text
        elif role != "system" and text is not None and prompt is not None:
            thoughts.append(text)

    if prompt is None:
        prompt = records[0].get(thought_key, "")
        thoughts = [r.get(thought_key, "") for r in records[1:]]

    if buffer_size is None:
        buffer_size = len(thoughts)

    random.seed(seed)
    np.random.seed(seed)

    model = SentenceTransformer(embed_model)
    vectors = model.encode([prompt] + thoughts, convert_to_numpy=True)
    prompt_vec = vectors[0]
    thought_vecs = vectors[1:]

    cos_to_prev: List[float] = []
    cos_to_prompt: List[float] = []
    for idx, vec in enumerate(thought_vecs):
        centered = vec - prompt_vec
        if idx == 0:
            cos_prev = np.nan
        else:
            prev_centered = thought_vecs[idx - 1] - prompt_vec
            cos_prev = float(cosine_similarity([centered], [prev_centered])[0][0])
        cos_prompt = float(cosine_similarity([vec], [prompt_vec])[0][0])
        cos_to_prev.append(cos_prev)
        cos_to_prompt.append(cos_prompt)

    df = pd.DataFrame(
        {
            "step": list(range(1, len(thoughts) + 1)),
            "buffer_size": buffer_size,
            "cos_to_prev": cos_to_prev,
            "cos_to_prompt": cos_to_prompt,
        }
    )
    return df
