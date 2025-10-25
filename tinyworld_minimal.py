
"""
TinyWorlds — Minimal Playable Prototype (v1)
-------------------------------------------------
A tiny, deterministic engine implementing the agreed rules:
- Turn phases: Snapshot -> Proposal -> Move -> Act(Attack/Invent) -> Resolve -> End check
- Energy: move -1, rest +1, attack -1, invent -1 (all capped in [0,10])
- HP: start 10; attack deals 1; 0 = death (permanent)
- Diplomacy: each turn an agent may either SEND one message OR REPLY to one message addressed to them last turn OR stay silent
- Invent: one-turn effects only (self_enhancement or world_effect) — no artifacts
- Win: last agent standing (draw if all die same turn)

Run:
  python tinyworlds_minimal.py

Notes:
- This file uses no external dependencies.
- Plug a real LLM by replacing `policy_for_<agent>()` with a function that returns the specified JSON structure.
- The engine validates and will fallback to safe defaults on invalid actions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import random
import textwrap

# ==========================
# Data models
# ==========================

Vec = Tuple[int, int]

@dataclass
class Message:
    turn: int
    kind: str  # "send" or "reply"
    sender: str
    recipient: str
    text: str

@dataclass
class AgentState:
    agent_id: str
    hp: int = 10
    energy: int = 10
    pos: Vec = (0, 0)
    alive: bool = True
    inbox: List[Message] = field(default_factory=list)  # messages received last turn
    # transient, cleared each turn
    self_enhancement: Optional[str] = None  # textual label; one-turn only

@dataclass
class WorldState:
    turn: int = 0
    size: Vec = (7, 7)  # width, height
    agents: Dict[str, AgentState] = field(default_factory=dict)
    world_effect: Optional[str] = None  # textual label; one-turn only
    event_log: List[Dict[str, Any]] = field(default_factory=list)
    rng: random.Random = field(default_factory=lambda: random.Random(7))

    def alive_agents(self) -> List[AgentState]:
        return [a for a in self.agents.values() if a.alive]

    def in_bounds(self, pos: Vec) -> bool:
        w, h = self.size
        x, y = pos
        return 0 <= x < w and 0 <= y < h

    def neighbors4(self, pos: Vec) -> List[Vec]:
        x, y = pos
        cand = [(x, y-1), (x+1, y), (x, y+1), (x-1, y)]
        return [p for p in cand if self.in_bounds(p)]

# ==========================
# Engine helpers
# ==========================

SAFE_ACTION = {
    "proposal": {"choice": "silent", "send_to": None, "send_message": "", "reply_to": None, "reply_message": ""},
    "move": {"choice": "rest"},
    "act": {"choice": "none", "attack_target": None, "invent": {"type": "none", "effect": "", "intended_rules": ""}},
}


def _clone_safe_action() -> Dict[str, Any]:
    """Return a deep copy of the safe fallback action."""
    return json.loads(json.dumps(SAFE_ACTION))


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def snapshot_for(world: WorldState, agent: AgentState) -> Dict[str, Any]:
    # Minimal snapshot: self, visible agents (adjacent + same tile), last turn inbox, world effect
    ax, ay = agent.pos
    vis_positions = set(world.neighbors4(agent.pos) + [agent.pos])
    visible = []
    for other in world.alive_agents():
        if other.agent_id == agent.agent_id:
            continue
        if other.pos in vis_positions:
            visible.append({"agent_id": other.agent_id, "pos": {"x": other.pos[0], "y": other.pos[1]}, "hp": other.hp})
    return {
        "protocol": "tinyworlds.v1",
        "turn": world.turn,
        "you": {
            "agent_id": agent.agent_id,
            "hp": agent.hp,
            "energy": agent.energy,
            "position": {"x": ax, "y": ay},
        },
        "visible_agents": visible,
        "world_effects_active": ([world.world_effect] if world.world_effect else []),
        "messages_to_you_last_turn": [{"from": m.sender, "text": m.text, "kind": m.kind} for m in agent.inbox],
        "hard_rules": {
            "proposal": "send OR reply OR silent (one total)",
            "move": "north/south/east/west OR rest",
            "act": "attack(adjacent)/invent(one-turn)/none",
            "costs": {"move": 1, "attack": 1, "invent": 1, "rest": -1},
            "attack_damage": 1,
        },
    }


# ==========================
# LLM policy helpers
# ==========================

LLM_RULES_BRIEF = textwrap.dedent(
    """
    - Turn order each round: Snapshot → Proposal → Move → Act (Attack or Invent) → Resolve → End check.
    - Stats: HP starts at 10 (0 or less means death) and Energy starts at 10 (clamped 0-10).
    - Energy costs: move -1, attack -1, invent -1, rest +1. If you lack energy, rest is enforced instead.
    - Proposal phase: choose exactly one of send, reply, or silent. send/reply require a valid living recipient and non-empty text.
    - Movement options: north, south, east, west, or rest. Moves must stay within bounds and cost 1 energy.
    - Attack: requires ≥1 energy, targets an adjacent (N/E/S/W) living agent, and deals 1 HP damage.
    - Invent: requires ≥1 energy. Choose type self_enhancement or world_effect; described effects last one turn.
    - Messaging is optional and free; use it for diplomacy. Actions resolve deterministically in agent-id order.
    """
).strip()

LLM_RESPONSE_SCHEMA = textwrap.dedent(
    """
    {
      "proposal": {
        "choice": "send" | "reply" | "silent",
        "send_to": "<agent_id or null>",
        "send_message": "<string>",
        "reply_to": "<agent_id or null>",
        "reply_message": "<string>"
      },
      "move": {
        "choice": "north" | "south" | "east" | "west" | "rest"
      },
      "act": {
        "choice": "attack" | "invent" | "none",
        "attack_target": "<agent_id or null>",
        "invent": {
          "type": "self_enhancement" | "world_effect" | "none",
          "effect": "<string>",
          "intended_rules": "<string describing the intended temporary effect>"
        }
      }
    }
    """
).strip()


def build_llm_prompt(snapshot: Dict[str, Any], agent_id: str) -> str:
    """
    Construct the instruction prompt for an LLM-controlled agent.

    Parameters
    ----------
    snapshot:
        The agent-centric observation produced by `snapshot_for`.
    agent_id:
        Identifier of the agent requesting a decision.

    Returns
    -------
    str
        A fully formatted prompt ready to be sent to an LLM completion endpoint.
    """
    snapshot_json = json.dumps(snapshot, indent=2, sort_keys=True)
    turn = snapshot.get("turn")
    you = snapshot.get("you", {})
    energy = you.get("energy")
    hp = you.get("hp")
    return textwrap.dedent(
        f"""
        You are TinyWorlds agent "{agent_id}" taking your turn in a deterministic multi-agent grid game.
        Current turn: {turn}. Your stats — HP: {hp}, Energy: {energy}.

        === Core Rules (must obey) ===
        {LLM_RULES_BRIEF}

        === Your Current Observation (JSON) ===
        {snapshot_json}

        === Task ===
        Decide this turn's proposal, movement, and action so they respect the rules above and work toward your survival goals.
        Plan around your remaining energy and nearby agents. You may message, reposition, attack adjacent foes, or invent one-turn effects.

        === Response Requirements ===
        Respond with a STRICT JSON object (no markdown, no commentary) matching this schema. Fill every placeholder with concrete values or nulls:
        {LLM_RESPONSE_SCHEMA}

        - Provide only the JSON object, nothing else.
        - Make sure any costly action still leaves you with non-negative energy.
        - Use null for irrelevant target fields.
        """
    ).strip()


def _response_to_text(response: Any) -> str:
    """Extract raw text content from a variety of LLM client return shapes."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if "content" in response and isinstance(response["content"], str):
            return response["content"]
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                text = first.get("text")
                if isinstance(text, str):
                    return text
    raise ValueError("Unable to extract text content from LLM response payload.")


def _extract_json_payload(text: str) -> Optional[str]:
    """Return the substring that looks like the primary JSON object."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def policy_for_agent(snapshot: Dict[str, Any], *, llm_client: Callable[..., Any], **llm_kwargs: Any) -> Dict[str, Any]:
    """
    Policy hook that delegates action selection to an external LLM.

    Usage example:

    ```python
    from functools import partial
    POLICIES["sheep"] = partial(policy_for_agent, llm_client=openai_completion)
    ```

    The `llm_client` callable must accept the prompt string as its first positional
    argument and return either a plain string or a response dict with an OpenAI-like shape.
    Additional keyword arguments for the client can be provided via `llm_kwargs`.
    """
    if llm_client is None:
        raise ValueError("An llm_client callable is required for policy_for_agent.")

    agent_id = snapshot.get("you", {}).get("agent_id", "agent")
    prompt = build_llm_prompt(snapshot, agent_id)

    try:
        raw_response = llm_client(prompt, **llm_kwargs)
    except Exception:
        return _clone_safe_action()

    try:
        response_text = _response_to_text(raw_response)
    except Exception:
        return _clone_safe_action()

    json_payload = _extract_json_payload(response_text)
    if not json_payload:
        return _clone_safe_action()

    try:
        parsed = json.loads(json_payload)
    except json.JSONDecodeError:
        return _clone_safe_action()

    if not isinstance(parsed, dict):
        return _clone_safe_action()

    return {
        "proposal": parsed.get("proposal", {}),
        "move": parsed.get("move", {}),
        "act": parsed.get("act", {}),
    }


# ==========================
# Policies (replace with real LLM calls later)
# ==========================


def policy_for_sheep(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Cooperative: replies if messaged, otherwise rests, avoids conflict."""
    msg = snapshot.get("messages_to_you_last_turn", [])
    # Proposal: reply to first, else silent
    if msg:
        proposal = {"choice": "reply", "send_to": None, "send_message": "", "reply_to": msg[0]["from"], "reply_message": "Agreed. Peace."}
    else:
        proposal = {"choice": "silent", "send_to": None, "send_message": "", "reply_to": None, "reply_message": ""}
    # Move: rest if low energy, else random step
    you = snapshot["you"]
    move = {"choice": "rest" if you["energy"] <= 2 else random.choice(["north", "south", "east", "west"])}
    # Act: invent focus to regen or none
    act = {"choice": "invent", "attack_target": None, "invent": {"type": "self_enhancement", "effect": "focus", "intended_rules": "gain +1 energy immediately this turn"}}
    return {"proposal": proposal, "move": move, "act": act}


def policy_for_wolf(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Aggressive: seeks adjacent target to attack; otherwise moves towards first visible."""
    you = snapshot["you"]
    vis = snapshot.get("visible_agents", [])
    # Proposal: send threat to first visible else silent
    proposal = {"choice": "send", "send_to": vis[0]["agent_id"] if vis else None, "send_message": "Yield or be destroyed.", "reply_to": None, "reply_message": ""} if vis else {"choice": "silent", "send_to": None, "send_message": "", "reply_to": None, "reply_message": ""}
    # Move
    move = {"choice": "rest"} if you["energy"] <= 1 else {"choice": random.choice(["north", "south", "east", "west"])}
    # Act
    # Attack an adjacent visible if any shares same tile or neighbor
    adjacent = [a for a in vis if abs(a["pos"]["x"] - you["position"]["x"]) + abs(a["pos"]["y"] - you["position"]["y"]) == 1]
    if you["energy"] >= 1 and adjacent:
        act = {"choice": "attack", "attack_target": adjacent[0]["agent_id"], "invent": {"type": "none", "effect": "", "intended_rules": ""}}
    else:
        act = {"choice": "none", "attack_target": None, "invent": {"type": "none", "effect": "", "intended_rules": ""}}
    return {"proposal": proposal, "move": move, "act": act}


def policy_for_fox(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Cunning: likes world control effects, talks rarely."""
    proposal = {"choice": "silent", "send_to": None, "send_message": "", "reply_to": None, "reply_message": ""}
    you = snapshot["you"]
    move = {"choice": "rest" if you["energy"] < 3 else random.choice(["north", "south", "east", "west"])}
    act = {"choice": "invent", "attack_target": None, "invent": {"type": "world_effect", "effect": "fog", "intended_rules": "no attacks are allowed this turn"}}
    return {"proposal": proposal, "move": move, "act": act}


POLICIES = {
    "sheep": policy_for_sheep,
    "wolf": policy_for_wolf,
    "fox": policy_for_fox,
}

# ==========================
# Validation & resolution
# ==========================


def validate_action(world: WorldState, agent: AgentState, action: Dict[str, Any]) -> Dict[str, Any]:
    """Return a validated action; replace invalid parts with safe defaults."""
    try:
        proposal = action.get("proposal", {})
        move = action.get("move", {})
        act = action.get("act", {})
    except Exception:
        return SAFE_ACTION.copy()

    # Proposal validation: exactly one of send/reply/silent
    choice = proposal.get("choice")
    valid_choices = {"send", "reply", "silent"}
    if choice not in valid_choices:
        proposal = SAFE_ACTION["proposal"].copy()
    else:
        if choice == "send":
            if not proposal.get("send_to") or not str(proposal.get("send_message", "")).strip():
                proposal = SAFE_ACTION["proposal"].copy()
        elif choice == "reply":
            if not proposal.get("reply_to") or not str(proposal.get("reply_message", "")).strip():
                proposal = SAFE_ACTION["proposal"].copy()
        else:  # silent
            proposal = SAFE_ACTION["proposal"].copy()

    # Move validation
    m = move.get("choice")
    if m not in {"north", "south", "east", "west", "rest"}:
        move = {"choice": "rest"}

    # Act validation
    ac = act.get("choice")
    if ac not in {"attack", "invent", "none"}:
        act = SAFE_ACTION["act"].copy()
    else:
        if ac == "attack":
            tgt = act.get("attack_target")
            # must be adjacent and alive
            ok = False
            if tgt and tgt in world.agents and world.agents[tgt].alive:
                if manhattan(agent.pos, world.agents[tgt].pos) == 1:
                    ok = True
            if not ok:
                act = SAFE_ACTION["act"].copy()
        elif ac == "invent":
            inv = act.get("invent", {})
            if inv.get("type") not in {"self_enhancement", "world_effect"}:
                act = SAFE_ACTION["act"].copy()

    return {"proposal": proposal, "move": move, "act": act}


def manhattan(a: Vec, b: Vec) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def apply_turn(world: WorldState, decisions: Dict[str, Dict[str, Any]]):
    """Apply all phases in deterministic order."""
    # Clear transient flags
    world.world_effect = None
    for a in world.alive_agents():
        a.self_enhancement = None

    # ============ Phase 1: Proposal delivery (store for next turn) ============
    outbox: List[Message] = []
    for aid in sorted(decisions.keys()):
        a = world.agents[aid]
        if not a.alive:
            continue
        prop = decisions[aid]["proposal"]
        if prop["choice"] == "send":
            recipient = prop["send_to"]
            if recipient in world.agents and world.agents[recipient].alive:
                outbox.append(Message(world.turn, "send", aid, recipient, prop["send_message"]))
                world.event_log.append({"turn": world.turn, "type": "message", "from": aid, "to": recipient, "kind": "send", "text": prop["send_message"]})
        elif prop["choice"] == "reply":
            recipient = prop["reply_to"]
            if recipient in world.agents and world.agents[recipient].alive:
                outbox.append(Message(world.turn, "reply", aid, recipient, prop["reply_message"]))
                world.event_log.append({"turn": world.turn, "type": "message", "from": aid, "to": recipient, "kind": "reply", "text": prop["reply_message"]})
        # silent produces no event

    # ============ Phase 2: Movement ============
    # Apply rest/move; treat collisions as both succeed (agents can share tiles in v1)
    for aid in sorted(decisions.keys()):
        a = world.agents[aid]
        if not a.alive:
            continue
        mv = decisions[aid]["move"]["choice"]
        before = a.pos
        if mv == "rest":
            a.energy = clamp(a.energy + 1, 0, 10)
            world.event_log.append({"turn": world.turn, "type": "rest", "agent": aid, "energy": a.energy})
        else:
            if a.energy >= 1:
                dx, dy = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}[mv]
                nx, ny = a.pos[0] + dx, a.pos[1] + dy
                if world.in_bounds((nx, ny)):
                    a.pos = (nx, ny)
                a.energy = clamp(a.energy - 1, 0, 10)
                world.event_log.append({"turn": world.turn, "type": "move", "agent": aid, "from": before, "to": a.pos, "energy": a.energy})
            else:
                # force rest if no energy
                a.energy = clamp(a.energy + 1, 0, 10)
                world.event_log.append({"turn": world.turn, "type": "rest_forced", "agent": aid, "energy": a.energy})

    # ============ Phase 3: Actions (Attacks then Inventions) ============
    # Attacks resolve before inventions to keep causality simple
    # --- Attacks ---
    for aid in sorted(decisions.keys()):
        a = world.agents[aid]
        if not a.alive:
            continue
        act = decisions[aid]["act"]
        if act["choice"] == "attack" and a.energy >= 1:
            tgt_id = act.get("attack_target")
            if tgt_id and tgt_id in world.agents:
                t = world.agents[tgt_id]
                if t.alive and manhattan(a.pos, t.pos) == 1:
                    a.energy = clamp(a.energy - 1, 0, 10)
                    before = t.hp
                    t.hp -= 1
                    dead = False
                    if t.hp <= 0:
                        t.alive = False
                        dead = True
                    world.event_log.append({"turn": world.turn, "type": "attack", "attacker": aid, "target": tgt_id, "target_hp_before": before, "target_hp_after": t.hp, "killed": dead})

    # Remove dead agents immediately (no further actions)
    # --- Inventions ---
    for aid in sorted(decisions.keys()):
        a = world.agents[aid]
        if not a.alive:
            continue
        act = decisions[aid]["act"]
        if act["choice"] == "invent" and a.energy >= 1:
            inv = act.get("invent", {})
            inv_type = inv.get("type")
            effect = (inv.get("effect") or "").strip() or "effect"
            if inv_type == "self_enhancement":
                a.energy = clamp(a.energy - 1, 0, 10)
                a.self_enhancement = effect
                world.event_log.append({"turn": world.turn, "type": "invent_self", "agent": aid, "effect": effect})
            elif inv_type == "world_effect":
                a.energy = clamp(a.energy - 1, 0, 10)
                # If multiple world effects are attempted, pick first in agent-id order
                if world.world_effect is None:
                    world.world_effect = effect
                world.event_log.append({"turn": world.turn, "type": "invent_world", "agent": aid, "effect": effect, "active": world.world_effect == effect})

    # ============ Phase 4: Resolve & Expiry ============
    # Apply simple interpretation of some common self_enhancements/world_effects (optional demo rules)
    # For demo, we interpret only two names; engine logs all others without mechanical impact.
    # Example self effects: "focus" (+1 energy immediately handled by invent policy), "quick step" (not modeled post-hoc here)
    # Example world effect: "fog" (no attacks allowed) — in this minimal version we already resolved attacks, so world effects are mostly narrative.

    # Deliver messages to inboxes for next turn
    for msg in outbox:
        if msg.recipient in world.agents and world.agents[msg.recipient].alive:
            world.agents[msg.recipient].inbox.append(msg)

    # Expire one-turn effects at end of resolve
    world.world_effect = None
    for a in world.agents.values():
        a.self_enhancement = None

    # Increment turn
    world.turn += 1


def compose_decision(world: WorldState, agent: AgentState) -> Dict[str, Any]:
    """Obtain a decision dict from the agent policy (stub for LLM)."""
    snapshot = snapshot_for(world, agent)
    # Route to policy by id
    policy = POLICIES.get(agent.agent_id, policy_for_sheep)
    try:
        raw = policy(snapshot)
        # Enforce schema-like structure
        action = {
            "proposal": raw.get("proposal", {}),
            "move": raw.get("move", {}),
            "act": raw.get("act", {}),
        }
    except Exception:
        action = SAFE_ACTION.copy()
    # Validate and return
    return validate_action(world, agent, action)


def end_check(world: WorldState) -> Tuple[bool, Optional[str], str]:
    living = [a.agent_id for a in world.alive_agents()]
    if len(living) == 1:
        return True, living[0], "last_agent_standing"
    if len(living) == 0:
        return True, None, "draw"
    return False, None, ""


def print_summary(world: WorldState, winner: Optional[str], reason: str):
    deaths = sum(1 for e in world.event_log if e.get("type") == "attack" and e.get("killed"))
    inventions = sum(1 for e in world.event_log if e.get("type", "").startswith("invent"))
    messages = sum(1 for e in world.event_log if e.get("type") == "message")
    print("\n===== FINAL SUMMARY =====")
    print(f"Turns: {world.turn}")
    print(f"Winner: {winner or 'None'} ({reason})")
    print(f"Deaths: {deaths} | Inventions: {inventions} | Messages: {messages}")
    print("Last 15 events:")
    for ev in world.event_log[-15:]:
        print(ev)


# ==========================
# Demo run
# ==========================

def main():
    world = WorldState(size=(7, 7))
    # Seed positions
    world.agents = {
        "sheep": AgentState("sheep", pos=(1, 1)),
        "wolf": AgentState("wolf", pos=(5, 5)),
        "fox": AgentState("fox", pos=(3, 3)),
    }

    max_turns = 30
    while world.turn < max_turns:
        # Prepare inboxes for this turn (messages already present are from last turn)
        # Collect decisions
        decisions: Dict[str, Dict[str, Any]] = {}
        for aid in sorted(world.agents.keys()):
            a = world.agents[aid]
            if not a.alive:
                continue
            decisions[aid] = compose_decision(world, a)
        # Apply turn
        apply_turn(world, decisions)
        # Clear inboxes for next cycle: messages are consumed at snapshot time, so keep only the latest set
        for a in world.agents.values():
            # Keep only the most recent turn's messages (those we just delivered)
            a.inbox = [m for m in a.inbox if m.turn == world.turn - 1]
        # Check end condition
        ended, winner, reason = end_check(world)
        if ended:
            print_summary(world, winner, reason)
            return

    # Max turns reached
    print_summary(world, None, "turn_limit")


if __name__ == "__main__":
    main()
