import datetime
import inspect
import json
import statistics
from typing import Any, Dict, List, Optional, Union
from autogoal.kb._algorithm import Algorithm, AlgorithmBase, Pipeline
from autogoal.meta_learning.utils import extract_algorithms_from_pipeline
from autogoal.search import Logger
from autogoal.meta_learning._experience import Experience, ExperienceStore, Metric
    
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
        # self.log_experience(
        #     solution=solution,
        #     metrics=None, # Changed
        #     error=e,
        # )
        pass

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
                    metric_key = f"{k}"
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
            task_features=self.dataset_features,
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