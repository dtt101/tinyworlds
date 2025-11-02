import random
from typing import Dict, List
from policies import Policy
from models import ALLOWED_MOVES 
from world import World


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
