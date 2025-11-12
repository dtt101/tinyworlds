import json

from ..models import Agent, ALLOWED_MOVES

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful planner. Always return valid JSON with a single key 'move' whose value is one of ['N','S','E','W','X'].\n"
    "Never include extra keys or commentary."
)

def build_llm_prompt(agent: Agent, world: "World") -> str:
    grid_ascii = world.render_ascii(highlight=agent.id)
    data = {
        "you": {
            "id": agent.id,
            "name": agent.name,
            "pos": agent.pos,
            "hp": agent.hp,
            "energy": agent.energy,
        },
        "others": [
            {"id": a.id, "name": a.name, "pos": a.pos, "hp": a.hp, "energy": a.energy}
            for a in world.agents if a.id != agent.id
        ],
        "board_size": world.size,
        "allowed_moves": ALLOWED_MOVES,
        "rules": (
            "Move exactly one square N/E/S/W (costs 1 energy) or rest (X) to stay put and regain 1 energy. "
            "If moving would leave you with negative energy you are forced to rest. Return strict JSON."
        ),
    }

    return (
        "You are an agent in a tiny grid world.\n" 
        "Current grid (your letter is uppercase):\n" + grid_ascii + "\n\n" 
        + json.dumps(data)
    )
