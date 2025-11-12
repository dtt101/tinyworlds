import random
from typing import Dict, List

from .policies import Policy
from .models import ALLOWED_MOVES
from .world import World


def step(world: World, policies: Dict[str, Policy], rng: random.Random) -> List[Dict]:
    """One round: each agent picks a move; engine applies sequentially."""
    events: List[Dict] = []
    for agent in world.agents:
        pol = policies[agent.id]
        requested_move = pol.choose_move(agent, world, rng)
        move = requested_move if requested_move in ALLOWED_MOVES else "X"

        note = "ok"
        if move != requested_move:
            note = "invalid_move_replaced_with_X"

        if move != "X" and agent.energy <= 0:
            move = "X"
            note = "forced_rest_low_energy"

        if move == "X":
            agent.energy += 1
            new_pos = agent.pos
            if note == "ok":
                note = "rest"
        else:
            agent.energy -= 1
            new_pos, move_note = world.move(agent.id, move)
            note = move_note

        events.append({
            "agent": agent.id,
            "name": agent.name,
            "move": move,
            "pos": new_pos,
            "note": note,
            "hp": agent.hp,
            "energy": agent.energy,
        })
    return events
