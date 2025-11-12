import random
from typing import Protocol

from ..models import Agent, ActionDecision

class Policy(Protocol):
    def choose_move(self, agent: Agent, world: "World", rng: random.Random) -> str: ...
    def choose_action(self, agent: Agent, world: "World", rng: random.Random) -> ActionDecision: ...
