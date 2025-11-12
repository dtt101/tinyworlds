from dataclasses import dataclass
from typing import Dict, Tuple

Vec = Tuple[int, int]

DIRS: Dict[str, Vec] = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0),
    "X": (0, 0),  # stay
}

ALLOWED_MOVES = list(DIRS.keys())

@dataclass
class Agent:
    id: str
    name: str
    pos: Vec
    hp: int = 10
    energy: int = 10

