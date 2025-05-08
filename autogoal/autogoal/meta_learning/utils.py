# Utility functions and helpers can be added here

from dataclasses import asdict, dataclass
from typing import Any

@dataclass
class Metric:
    name: str
    maximize: bool
    value: float

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            maximize=bool(d["maximize"]),
            value=float(d["value"]),
        )
        
@dataclass
class MetricSpec:
    """
    Specification for a metric used in meta-learning.
    Only name and optionally weight are required for user input.
    The maximize flag is inferred from the experiences.
    """
    name: str
    weight: float = 1.0
    maximize: bool = True  # Always set internally, not required from user

    def to_metric(self, value: Any) -> Metric:
        return Metric(name=self.name, maximize=self.maximize, value=float(value))

    @classmethod
    def from_dict(cls, d: dict) -> "MetricSpec":
        """
        Create a MetricSpec from a dictionary, using defaults for missing fields.
        Only 'name' and optionally 'weight' are required from user.
        """
        return cls(
            name=d.get("name"),
            weight=d.get("weight", 1.0),
        )
    