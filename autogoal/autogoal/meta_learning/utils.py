# Utility functions and helpers can be added here

from dataclasses import asdict, dataclass
import inspect
import json
from typing import Any, Dict, List, Optional

from autogoal.kb._algorithm import Algorithm, AlgorithmBase, Pipeline


def extract_algorithms_from_pipeline(pipeline: Pipeline) -> List[Dict[str, Any]]:
    """
    Extracts algorithms and their parameters from the given pipeline in a recursive manner,
    capturing only the parameters defined in each algorithm's __init__ method.

    Parameters:
        pipeline (Pipeline): The pipeline object to extract information from.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries where each dictionary has the algorithm class name
                              as the key and its sanitized parameters as the value, recursively.
    """
    if pipeline is None:
        return []

    algorithms_info = []

    for alg in pipeline.algorithms:
        alg_info = extract_algorithm_info(alg)
        algorithms_info.append(alg_info)

    return algorithms_info


def extract_algorithm_info(alg: Algorithm) -> Dict[str, Any]:
    """
    Recursively extracts information from an algorithm, capturing only the parameters defined in its __init__ method.

    Parameters:
        alg (Algorithm): The algorithm to extract information from.

    Returns:
        Dict[str, Any]: A dictionary with the algorithm class name as key and its sanitized parameters as value.
    """
    class_name = alg.__class__.__name__
    parameters = {}

    # Retrieve the signature of the __init__ method
    signature = alg.__class__.get_inner_signature()
    for param_name, param_obj in signature.parameters.items():
        if param_name in ["self", "args", "kwargs"]:
            continue

        annotation_cls = param_obj.annotation
        if annotation_cls == inspect.Parameter.empty:
            continue

        value = getattr(alg, param_name, None)
        sanitized_value = sanitize_for_json(value)
        if sanitized_value is not None:
            if hasattr(annotation_cls, "__name__"):
                parameters[param_name] = {
                    "annotation": annotation_cls.__name__,
                    "value": sanitized_value,
                }
            else:
                parameters[param_name] = {
                    "value": sanitized_value,
                }
        else:
            pass

    return {class_name: parameters}


def sanitize_for_json(value):
    """
    Recursively sanitizes a value to ensure it is JSON serializable.

    Parameters:
        value: The value to sanitize.

    Returns:
        A JSON-serializable version of the value.
    """
    import numpy as np

    if value is None:
        return None
    # Handle numpy arrays first
    elif isinstance(value, np.ndarray):
        return value.tolist()  # Convert ndarray to list
    # Handle numpy scalar types
    elif isinstance(value, (np.generic,)):
        if isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        else:
            return value.item()
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    elif isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    elif isinstance(value, (AlgorithmBase, Pipeline)):
        # For algorithms and pipelines, extract their info recursively
        return (
            extract_algorithm_info(value)
            if isinstance(value, Algorithm)
            else extract_algorithms_from_pipeline(value)
        )
    else:
        # Attempt to serialize; if fails, exclude the value
        try:
            json.dumps(value)
            return value
        except TypeError:
            # Exclude the value
            return None


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
    best_value: Optional[float] = None  # Ideal or target value for this metric

    def to_metric(self, value: Any) -> Metric:
        return Metric(name=self.name, maximize=self.maximize, value=float(value))

    @classmethod
    def from_dict(cls, d: dict) -> "MetricSpec":
        """
        Create a MetricSpec from a dictionary, using defaults for missing fields.
        Only 'name' and optionally 'weight' and 'best_value' are required from user.
        """
        if not d.get("name"):
            raise ValueError("MetricSpec requires a 'name' field in from_dict.")
        return cls(
            name=d.get("name"),
            weight=d.get("weight", 1.0),
            best_value=d.get("best_value"),  # Add best_value here
        )
