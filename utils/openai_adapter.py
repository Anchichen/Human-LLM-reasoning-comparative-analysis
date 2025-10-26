from __future__ import annotations
from typing import List, Optional
from types import SimpleNamespace
import openai

def _to_chat_shape(resp: openai.resources.responses.Response) -> SimpleNamespace:
    """
    Convert a /responses result into a tiny object that looks like a
    chat-completions response:  response.choices[0].message.content
    and   response.usage.prompt_tokens / completion_tokens.
    """
    # Pull every text chunk from the first textual output
    text_chunks = [
        piece.text
        for block in resp.output
        if block.type == "text"
        for piece in block.content
        if piece.type == "text"
    ]
    full_text = "".join(text_chunks)

    # Build the minimal structure BufferedLLM expects
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=full_text))],
        # usage isn't returned by /responses yet – stub zeros
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
        _raw=resp,      # keep a reference to the real object if you need it
    )

def _prompt_from_messages(messages: List[dict]) -> str:
    """Flatten chat-style messages → plain text."""
    return "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in messages
    )

"""A tiny wrapper that chooses the right OpenAI endpoint.

Usage
-----
from _shared import _create
response = _create(client, model_name, messages=messages, temperature=0.7)
"""

def create_completion(
    client: openai.OpenAI,
    model: str,
    *,
    messages: Optional[List[dict]] = None,
    input: Optional[str] = None,
    **kwargs,
) -> SimpleNamespace | openai.OpenAIObject:
    """
    Dispatch:
      - If `messages` is provided, use /v1/chat/completions (all models).
      - Otherwise, use /v1/completions with only safe kwargs.
    """
    if messages is not None:
        # —— Chat-style invocation ————————————————
        if model.startswith("o"):
            # o-series chat models only support the default sampling params
            banned = {
                "temperature",
                "top_p",
                "presence_penalty",
                "frequency_penalty",
            }
            safe_kwargs = {k: v for k, v in kwargs.items() if k not in banned}
            return client.chat.completions.create(
                model=model,
                messages=messages,
                **safe_kwargs,
            )

        # All other chat models (gpt-3.5, gpt-4o, etc.)
        return client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )