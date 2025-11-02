"""
TinyWorlds: Where AIs compete
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Protocol, Optional
import argparse
import random
import json
import os
from dotenv import load_dotenv
from openai import OpenAI  # type: ignore

from models import Agent, ALLOWED_MOVES 
from world import World

load_dotenv()

class Policy(Protocol):
    def choose_move(self, agent: Agent, world: "World", rng: random.Random) -> str: ...

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

def build_llm_prompt(agent: Agent, world: "World") -> str:
    grid_ascii = world.render_ascii(highlight=agent.id)
    data = {
        "you": {"id": agent.id, "name": agent.name, "pos": agent.pos},
        "others": [
            {"id": a.id, "name": a.name, "pos": a.pos}
            for a in world.agents if a.id != agent.id
        ],
        "board_size": world.size,
        "allowed_moves": ALLOWED_MOVES,
        "rules": "You may move exactly one square N/E/S/W or stay (X). Return strict JSON.",
    }
    # Human-readable + machine-readable
    return (
        "You are an agent in a tiny grid world.\n" 
        "Current grid (your letter is uppercase):\n" + grid_ascii + "\n\n" 
        + json.dumps(data)
    )

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful planner. Always return valid JSON with a single key 'move' whose value is one of ['N','S','E','W','X'].\n"
    "Never include extra keys or commentary."
)


def step(world: World, policies: Dict[str, Policy], rng: random.Random) -> List[Dict]:
    """One round: each agent picks a move; engine applies sequentially."""
    events: List[Dict] = []
    for agent in world.agents:
        pol = policies[agent.id]
        move = pol.choose_move(agent, world, rng)
        if move not in ALLOWED_MOVES:
            move = "X"
            note = "invalid_move_replaced_with_X"
            new_pos = agent.pos
        else:
            new_pos, note = world.move(agent.id, move)
        events.append({
            "agent": agent.id,
            "name": agent.name,
            "move": move,
            "pos": new_pos,
            "note": note,
        })
    return events


def demo(size: int, rounds: int, seed: int) -> None:
    rng = random.Random(seed)
    agents = [
        Agent(id="a1", name="alice", pos=(0, 0)),
        Agent(id="b2", name="bob",   pos=(size - 1, size - 1)),
    ]
    world = World(size=size, agents=agents)
    policies: Dict[str, Policy] = {
        "a1": LLMPolicy(),
        "b2": LLMPolicy(),
    }

    print("Start:\n" + world.render_ascii())
    for r in range(1, rounds + 1):
        events = step(world, policies, rng)
        print(f"\nRound {r}")
        print(world.render_ascii())
        print("events:", json.dumps(events))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    demo(size=args.size, rounds=args.rounds, seed=args.seed)

if __name__ == "__main__":
    main()

