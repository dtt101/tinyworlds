"""
Command-line entrypoints for TinyWorlds demos.
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv() -> None:
        return None

from .engine import step
from .models import Agent
from .policies import LLMPolicy, Policy
from .world import World


def run_demo(*, size: int, rounds: int, seed: int) -> None:
    """Run the original proof-of-concept demo with two LLM-backed agents."""
    rng = random.Random(seed)
    agents = [
        Agent(id="a1", name="alice", pos=(0, 0)),
        Agent(id="b2", name="bob", pos=(size - 1, size - 1)),
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


def main(argv: Optional[list[str]] = None) -> int:
    """Parse CLI arguments and execute the demo."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    load_dotenv()
    run_demo(size=args.size, rounds=args.rounds, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
