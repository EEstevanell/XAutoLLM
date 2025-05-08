
from dataclasses import dataclass
from typing import Any, Callable

@dataclass(frozen=True)
class Objective:
    name: str
    metric: Callable[[Any, Any], float]
    maximize: bool = True