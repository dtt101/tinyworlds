"""
TinyWorlds: Where AIs compete
"""
from __future__ import annotations
from typing import Dict, List 
import argparse
import random
import json
import os
from dotenv import load_dotenv

from models import Agent, ALLOWED_MOVES 
from world import World
from policies import LLMPolicy 

load_dotenv()

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

