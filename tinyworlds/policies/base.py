import random
from typing import Protocol

from ..models import Agent

class Policy(Protocol):
    def choose_move(self, agent: Agent, world: "World", rng: random.Random) -> str: ...
