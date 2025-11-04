from typing import Dict, List, Tuple, Optional

from .models import Agent, Vec, DIRS

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
