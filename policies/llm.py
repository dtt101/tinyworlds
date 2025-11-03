import random
import json
from typing import Optional 
from models import Agent, ALLOWED_MOVES 
from world import World
from .prompts import DEFAULT_SYSTEM_PROMPT, build_llm_prompt

from openai import OpenAI  # type: ignore

class LLMPolicy:
    """
    OpenAI-backed policy. Requires OPENAI_API_KEY env var. Optional model via
    TINYWORLDS_OPENAI_MODEL (default: 'gpt-4.1-mini'). Uses JSON mode and
    strict parsing; falls back to 'X' on any error.
    """
    def __init__(self, system_prompt: str | None = None, model: Optional[str] = None):
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.model = model
        self._client_cls = OpenAI

    def _pick_model(self) -> str:
        import os
        return self.model or os.getenv("TINYWORLDS_OPENAI_MODEL", "gpt-4.1-mini")

    def _call_openai(self, prompt: str) -> str:
        client = self._client_cls()
        resp = client.responses.create(
            model=self._pick_model(),
            input=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        try:
            return resp.output_text  # type: ignore[attr-defined]
        except Exception:
            if hasattr(resp, "output") and resp.output and len(resp.output) > 0:
                part = resp.output[0]
                content = getattr(part, "content", None)
                if content and len(content) > 0:
                    return getattr(content[0], "text", "")
            return ""

    def choose_move(self, agent: Agent, world: "World", rng: random.Random) -> str:
        prompt = build_llm_prompt(agent, world)
        raw = ""
        for _ in range(2):  # one retry on parse failure
            raw = self._call_openai(prompt)
            obj = json.loads(raw)
            mv = obj.get("move")
            if isinstance(mv, str) and mv in ALLOWED_MOVES:
                return mv
            prompt = prompt + "Reminder: Return only {\"move\": \"N|S|E|W|X\"}."
        return "X"
