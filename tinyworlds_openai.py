"""
OpenAI policy integration for TinyWorlds.

This module exposes helpers that adapt the generic `policy_for_agent`
function defined in `tinyworld_minimal.py` to the OpenAI Python SDK.
The resulting callable can be plugged into the engine's `POLICIES`
mapping to drive agents with live model output.

Example usage:

```python
from tinyworlds_openai import make_openai_policy
from tinyworld_minimal import POLICIES

POLICIES["sheep"] = make_openai_policy(model="gpt-4o-mini")
```

Ensure the `OPENAI_API_KEY` environment variable is set or provide the
`api_key` argument when constructing the policy.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Optional

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "The openai package is required"
    ) from exc

from tinyworld_minimal import policy_for_agent


def _make_openai_call(
    *,
    client: OpenAI,
    model: str,
    temperature: float,
    max_output_tokens: Optional[int],
) -> Callable[[str], Any]:
    """
    Wrap the OpenAI client in the signature expected by `policy_for_agent`.

    The wrapper sends a single-user chat completion request containing the
    prepared prompt and returns the raw API response, which
    `policy_for_agent` will parse.
    """

    def _invoke(prompt: str, **extra: Any) -> Any:
        messages = [
            {
                "role": "system",
                "content": (
                    "You output only strict JSON that encodes agent actions for the TinyWorlds engine."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        payload.update(extra)
        return client.chat.completions.create(**payload)

    return _invoke


def make_openai_policy(
    *,
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 0.4,
    max_output_tokens: Optional[int] = 512,
    client: Optional[OpenAI] = None,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Create a TinyWorlds agent policy that queries an OpenAI chat model.

    Parameters
    ----------
    model:
        The OpenAI chat model ID to use (e.g., "gpt-4o-mini").
    api_key:
        Optional API key override. Defaults to reading from environment.
    temperature:
        Sampling temperature passed to the chat completion API.
    max_output_tokens:
        Upper bound on tokens generated for the JSON response. Set to None to rely on server defaults.
    client:
        Optional preconstructed `OpenAI` client. When provided, `api_key` is ignored.
    """
    openai_client = client or OpenAI(api_key=api_key)
    llm_client = _make_openai_call(
        client=openai_client,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return partial(policy_for_agent, llm_client=llm_client)

