from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal, Any

Vec = Tuple[int, int]

DIRS: Dict[str, Vec] = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0),
    "X": (0, 0),  # stay
}

ALLOWED_MOVES = list(DIRS.keys())

ALLOWED_ACTIONS = ["attack", "none"]

@dataclass
class Agent:
    id: str
    name: str
    pos: Vec
    hp: int = 10
    energy: int = 10

ActionType = Literal["attack", "none"]

@dataclass
class ActionDecision:
    """Normalized representation of an action request from a policy."""
    kind: ActionType = "none"
    target: Optional[str] = None

    @classmethod
    def from_request(cls, raw: Any) -> "ActionDecision":
        """Coerce arbitrary policy output into a valid ActionDecision."""
        if isinstance(raw, cls):
            return raw

        if isinstance(raw, str):
            val = raw.strip().lower()
            if val == "attack":
                return cls(kind="attack")
            return cls()

        if isinstance(raw, dict):
            # Accept both 'kind' and 'type' for ergonomics
            kind = raw.get("kind") or raw.get("type") or "none"
            target = raw.get("target")
            if isinstance(kind, str):
                kind = kind.lower()
            else:
                kind = "none"
            if kind not in ALLOWED_ACTIONS:
                kind = "none"
                target = None
            if target is not None:
                target = str(target)
            return cls(kind=kind, target=target)

        return cls()

def is_adjacent(a: Vec, b: Vec) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
