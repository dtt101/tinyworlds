import random
from typing import Dict, List, Set

from .policies import Policy
from .models import ALLOWED_MOVES, ActionDecision, is_adjacent
from .world import World


def step(world: World, policies: Dict[str, Policy], rng: random.Random) -> List[Dict]:
    """Process one full round of phases (move + action) and return structured events."""
    events: List[Dict] = []
    # Move phase: resolve sequentially in ID order
    for agent in list(world.agents):
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
            "phase": "move",
            "agent": agent.id,
            "name": agent.name,
            "move": move,
            "pos": new_pos,
            "note": note,
            "hp": agent.hp,
            "energy": agent.energy,
        })

    # Action phase: currently only supports attack/none
    to_remove: Set[str] = set()
    for agent in list(world.agents):
        pol = policies[agent.id]
        raw_action = pol.choose_action(agent, world, rng)
        action = ActionDecision.from_request(raw_action)
        note = "idle"
        target_id = action.target
        target_name = None
        target_hp = None

        if agent.energy <= 0:
            note = "forced_none_low_energy"
            action = ActionDecision()
            target_id = None
        elif action.kind == "attack":
            if agent.energy - 1 < 0:
                note = "forced_none_low_energy"
                action = ActionDecision()
                target_id = None
            elif target_id is None:
                note = "invalid_attack_no_target"
            else:
                try:
                    target_agent = world.get(target_id)
                except KeyError:
                    note = "invalid_attack_unknown_target"
                else:
                    if target_agent.hp <= 0:
                        note = "invalid_attack_target_dead"
                    elif not is_adjacent(agent.pos, target_agent.pos):
                        note = "invalid_attack_not_adjacent"
                    else:
                        agent.energy -= 1
                        target_agent.hp -= 1
                        target_name = target_agent.name
                        target_hp = target_agent.hp
                        note = "attack_hit"
                        if target_agent.hp <= 0:
                            note = "attack_kill"
                            to_remove.add(target_agent.id)
        else:
            note = "idle"

        events.append({
            "phase": "action",
            "agent": agent.id,
            "name": agent.name,
            "action": action.kind,
            "target": target_id,
            "target_name": target_name,
            "target_hp": target_hp,
            "note": note,
            "hp": agent.hp,
            "energy": agent.energy,
        })

    # Resolve deaths after all actions so order is deterministic
    for dead_id in sorted(to_remove):
        try:
            dead_agent = world.get(dead_id)
        except KeyError:
            continue
        death_pos = dead_agent.pos
        world.remove(dead_id)
        events.append({
            "phase": "resolve",
            "agent": dead_id,
            "name": dead_agent.name,
            "pos": death_pos,
            "note": "removed_dead",
            "hp": dead_agent.hp,
            "energy": dead_agent.energy,
        })
    return events
