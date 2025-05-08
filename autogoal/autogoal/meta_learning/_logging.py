import datetime
import inspect
import json
import statistics
from typing import Any, Dict, List, Optional, Union
from autogoal.kb._algorithm import Algorithm, AlgorithmBase, Pipeline
from autogoal.search import Logger
from autogoal.meta_learning._experience import Experience, ExperienceStore, Metric

def sanitize_for_json(value):
    """
    Recursively sanitizes a value to ensure it is JSON serializable.
    
    Parameters:
        value: The value to sanitize.
    
    Returns:
        A JSON-serializable version of the value.
    """
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    elif isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    elif isinstance(value, (AlgorithmBase, Pipeline)):
        # For algorithms and pipelines, extract their info recursively
        return extract_algorithm_info(value) if isinstance(value, Algorithm) else extract_algorithms_from_pipeline(value)
    else:
        # Attempt to serialize; if fails, exclude the value
        try:
            json.dumps(value)
            return value
        except TypeError:
            # Exclude the value
            return None
    
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
        
    return { class_name: parameters }


class ExperienceLogger(Logger):
    def __init__(
        self,
        dataset_features=None,
        system_features=None,
        dataset_feature_extractor_name=None,
        system_feature_extractor_name=None,
        alias: str = 'default',
    ) -> None:
        self.dataset_features = dataset_features
        self.system_features = system_features
        self.dataset_feature_extractor_name = dataset_feature_extractor_name
        self.system_feature_extractor_name = system_feature_extractor_name
        self.alias = alias

    def begin(self, generations, pop_size):
        pass

    def start_generation(self, generations, best_solutions, best_fns):
        pass

    def update_best(
        self,
        solution,
        fn,
        new_best_solutions,
        best_solutions,
        new_best_fns,
        best_fns,
        new_dominated_solutions,
    ):
        pass

    def error(self, e: Exception, solution, objective_names=None, objective_maximize=None):
        # Log the experience with error information
        self.log_experience(
            solution=solution,
            metrics=None, # Changed
            error=e,
        )

    def eval_solution(self, solution, fitness, observations, objective_names, objective_maximize):
        metrics: list = []
        current_fitness_values = fitness if isinstance(fitness, (list, tuple)) else [fitness]
        names = list(objective_names) if isinstance(objective_names, (list, tuple)) else [f"objective_{i}" for i in range(len(current_fitness_values))]
        flags = list(objective_maximize) if isinstance(objective_maximize, (list, tuple)) else [True] * len(current_fitness_values)
        if len(names) < len(current_fitness_values): names += [f"objective_{i}" for i in range(len(names), len(current_fitness_values))]
        if len(flags) < len(current_fitness_values): flags += [True] * (len(current_fitness_values) - len(flags))
        names, flags = names[:len(current_fitness_values)], flags[:len(current_fitness_values)]

        for name, flag, val in zip(names, flags, current_fitness_values):
            try:
                metrics.append(Metric(name=name, maximize=bool(flag), value=float(val)))
            except (ValueError, TypeError):
                print(f"ExperienceLogger Warning: Could not convert fitness value '{val}' for metric '{name}' to float.")

        if observations is not None:
            if "time" in observations and isinstance(observations["time"], dict):
                for time_type in ["train", "valid"]:
                    if time_type in observations["time"] and observations["time"][time_type]:
                        try:
                            numeric_times = [t for t in observations["time"][time_type] if isinstance(t, (int, float))]
                            if numeric_times:
                                metrics.append(Metric(
                                    name=f"time_{time_type}_mean_seconds",
                                    maximize=False,
                                    value=float(statistics.mean(numeric_times))
                                ))
                        except Exception as e:
                            print(f"ExperienceLogger Warning: Could not process time observation for {time_type}: {e}")
            for k, v_obs in observations.items():
                if k != "time" and isinstance(v_obs, (int, float)):
                    metric_key = f"obs_{k}"
                    if not any(m.name == metric_key for m in metrics):
                        metrics.append(Metric(name=metric_key, maximize=True, value=float(v_obs)))
                    elif not any(m.name == k for m in metrics):
                        metrics.append(Metric(name=k, maximize=True, value=float(v_obs)))

        self.log_experience(
            solution=solution,
            metrics=metrics if metrics else None,
            error=None,
        )

    def end(self, best_solutions, best_fns):
        pass

    def append_scores(self, scores, best_solutions):
        pass

    def log_experience(
        self,
        solution,
        metrics: Optional[list],
        error: Optional[Exception] = None,
    ):
        if solution is None:
            return

        algorithms_info = extract_algorithms_from_pipeline(solution)
        if not algorithms_info:
            return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ensure metrics is a list of Metric objects
        metric_objs = []
        if metrics is not None:
            for m in metrics:
                if isinstance(m, Metric):
                    metric_objs.append(m)
                elif isinstance(m, dict):
                    metric_objs.append(Metric(**m))
        experience = Experience(
            algorithms=algorithms_info,
            dataset_features=self.dataset_features,
            system_features=self.system_features,
            dataset_feature_extractor_name=self.dataset_feature_extractor_name,
            system_feature_extractor_name=self.system_feature_extractor_name,
            timestamp=timestamp,
            alias=self.alias,
            cross_val_steps=None,
            metrics=metric_objs,
        )

        if error is not None:
            experience.error = str(error)
            experience.metrics = []

        ExperienceStore.save_experience(experience)