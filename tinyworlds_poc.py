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

from models import Agent, ALLOWED_MOVES, DIRS

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

class World:
    def __init__(self, size: int, agents: List[Agent]):
        self.size = size
        # Sorted by id for deterministic turn order
        self.agents: List[Agent] = sorted(agents, key=lambda a: a.id)
        # Fast occupancy lookup
        self.occ: Dict[Vec, str] = {a.pos: a.id for a in self.agents}

    def in_bounds(self, pos: Vec) -> bool:
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size

    def is_free(self, pos: Vec) -> bool:
        return pos not in self.occ

    def move(self, agent_id: str, direction: str) -> Tuple[Vec, str]:
        agent = self.get(agent_id)
        dx, dy = DIRS.get(direction, (0, 0))
        target = (agent.pos[0] + dx, agent.pos[1] + dy)
        if not self.in_bounds(target):
            return agent.pos, f"blocked: out_of_bounds {target}"
        if direction != "X" and not self.is_free(target):
            return agent.pos, f"blocked: occupied {target}"
        # apply move
        del self.occ[agent.pos]
        agent.pos = target
        self.occ[target] = agent.id
        return agent.pos, "ok"

    def get(self, agent_id: str) -> Agent:
        for a in self.agents:
            if a.id == agent_id:
                return a
        raise KeyError(agent_id)

    def render_ascii(self, highlight: Optional[str] = None) -> str:
        # Represent each agent as first letter of name (uppercase for highlight)
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        for a in self.agents:
            x, y = a.pos
            ch = (a.name[:1] or "?").lower()
            if highlight and a.id == highlight:
                ch = ch.upper()
            grid[y][x] = ch
        return "\n".join("".join(row) for row in grid)


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

