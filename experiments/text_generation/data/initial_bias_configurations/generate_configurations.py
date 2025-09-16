from pathlib import Path
from typing import Dict, List, Optional
from autogoal.datasets import drop, squad
from autogoal.kb._semantics import GeneratedText, Prompt, Seq, Supervised
from autogoal.meta_learning.feature_extraction.generative_task import (
    GenerativeTaskFeatureExtractor,
)
from autogoal.meta_learning.warm_start import (
    WarmStart,
    EuclideanDistance,
    CosineDistance,
)
from autogoal.meta_learning.normalization import LogNormalizer, MinMaxNormalizer
from autogoal.ml import AutoML
import json
import random
from autogoal.ml.metrics import evaluation_time
from autogoal.search._warm_start_pge import NSPEWarmStartSearch
from autogoal.utils import Gb, Hour
from autogoal_contrib import find_classes
from autogoal_transformers._manual import (
    FineTuneGenLLMTask,
    LoraGenLLMTask,
    PartialFineTuneGenLLMTask,
)
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

FEATURE_CACHE_JSON_PATH = Path(
    "/home/coder/autogoal/experiments/text_generation/src/task_features_cache"
)


def _serialize_task_features(
    task_features: Dict[str, Optional[np.ndarray]],
) -> Dict[str, Optional[List[float]]]:
    """Converts np.ndarray features in the task features dictionary to lists for JSON serialization."""
    if task_features is None:
        return None
    serialized = {}
    for key, value in task_features.items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif value is None:
            serialized[key] = None
        else:
            # Should ideally not happen if type hints are followed, but good for robustness
            serialized[key] = list(value) if isinstance(value, (list, tuple)) else value
    return serialized


def _deserialize_task_features(
    loaded_features: Dict[str, Optional[List[float]]],
) -> Dict[str, Optional[np.ndarray]]:
    """Converts list features in a dictionary loaded from JSON back to np.ndarray."""
    if loaded_features is None:
        return None
    deserialized = {}
    for key, value in loaded_features.items():
        if isinstance(value, list):
            deserialized[key] = np.array(
                value, dtype=float
            )  # Assuming float for features
        elif value is None:
            deserialized[key] = None
        else:
            # Should ideally not happen
            deserialized[key] = (
                np.array(value, dtype=float)
                if isinstance(value, (list, tuple))
                else value
            )
    return deserialized


def log_model_state(model, dataset_name, config_name):
    # Log the distributions for generative LLM tasks
    model_info = {}
    for key in [
        "FineTuneGenLLMTask",
        "LoraGenLLMTask",
        "PartialFineTuneGenLLMTask",
    ]:
        if key in model and hasattr(model[key], "value"):
            model_info[key] = model[key].value

    path = f"autogoal/experiments/text_generation/data/initial_bias_configurations/{dataset_name}/{config_name}.json"
    try:
        # Create file only if it doesn't already exist
        with open(path, "x") as f:
            json.dump(model_info, f, indent=4)
    except FileExistsError:
        # If it exists, open in write mode to truncate/overwrite
        with open(path, "w") as f:
            json.dump(model_info, f, indent=4)
    except Exception as e:
        logger.error(f"Error logging model state for {config_name}: {e}")


def main():
    from autogoal.utils._process import initialize_cuda_multiprocessing

    initialize_cuda_multiprocessing()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    datasets = {"squad": squad, "drop": drop}
    dataset_options = ["drop", "squad"]

    distances = {
        "euc": EuclideanDistance,
        "cos": CosineDistance,
    }
    distances_options = ["euc", "cos"]

    k_pos_options = [0, 10000]
    k_neg_options = [0, 10000]

    adaptative_pos_options = [None, 1]
    adaptative_neg_options = [None, -1]

    utility_function_options = ["weighted_sum", "linear_front", "logarithmic_front"]
    beta_scale_options = [0, 0.5, 1.0]

    for dataset_name in dataset_options:
        X_train, y_train, X_test, y_test = datasets[dataset_name].load(True)

        for distance in distances_options:
            for k_pos in k_pos_options:
                for k_neg in k_neg_options:
                    for adaptative_pos in adaptative_pos_options:
                        for adaptative_neg in adaptative_neg_options:
                            for utility in utility_function_options:
                                for beta_scale in beta_scale_options:
                                    if k_pos == 0 and k_neg == 0:
                                        continue

                                    k_pos_name = (
                                        "no-pos"
                                        if k_pos == 0
                                        else (
                                            "f-pos"
                                            if adaptative_pos is None
                                            else "a-pos"
                                        )
                                    )
                                    k_neg_name = (
                                        "no-neg"
                                        if k_neg == 0
                                        else (
                                            "f-neg"
                                            if adaptative_neg is None
                                            else "a-neg"
                                        )
                                    )
                                    distance_name = (
                                        ""
                                        if beta_scale == 0
                                        else (
                                            f"cos k({beta_scale}) "
                                            if distance == "cos"
                                            else f"euc k({beta_scale}) "
                                        )
                                    )
                                    config_name = f"{dataset_name} - {k_pos_name} {k_neg_name} {distance_name}- utility ({utility})"

                                    ws = WarmStart(
                                        dataset_feature_extractor=GenerativeTaskFeatureExtractor,
                                        positive_min_threshold=0.2,
                                        distance=distances[distance],
                                        k_pos=k_pos,
                                        k_neg=k_neg,
                                        normalizers=[
                                            LogNormalizer(),
                                            MinMaxNormalizer(),
                                        ],
                                        adaptative_positive_alpha_limit=adaptative_pos,
                                        adaptative_negative_alpha_limit=adaptative_neg,
                                        utility_function=utility,
                                        beta_scale=beta_scale,
                                        max_alpha=0.05,
                                        min_alpha=-0.02,
                                        exclude=dataset_name,
                                        exit_after_warmup=True,
                                        on_warmup_exit=lambda model: log_model_state(
                                            model,
                                            dataset_name=dataset_name,
                                            config_name=config_name,
                                        ),
                                    )

                                    feature_cache_json_path = (
                                        FEATURE_CACHE_JSON_PATH / f"{dataset_name}.json"
                                    )

                                    current_task_features = None
                                    try:
                                        with open(feature_cache_json_path, "r") as f:
                                            loaded_f = json.load(f)
                                            current_task_features = (
                                                _deserialize_task_features(loaded_f)
                                            )
                                    except (
                                        Exception
                                    ) as e:  # Catch other potential errors
                                        logger.error(
                                            f"Error loading features from {feature_cache_json_path}: {e}"
                                        )
                                        current_task_features = None

                                    # initialize the WarmStart object with the current task features (may be None)
                                    ws.pre_warm_up(
                                        X_train,
                                        y_train,
                                        current_task_features=current_task_features,
                                    )

                                    objectives = [
                                        {
                                            "name": "f1",
                                            "metric": evaluation_time,
                                            "maximize": True,
                                        },
                                        {
                                            "name": "evaluation_time",
                                            "metric": evaluation_time,
                                            "maximize": False,
                                        },
                                    ]

                                    algorithm_registry = [
                                        FineTuneGenLLMTask,
                                        LoraGenLLMTask,
                                        PartialFineTuneGenLLMTask,
                                    ] + find_classes(include="TEXT_GEN")

                                    try:
                                        ml = AutoML(
                                            input=(
                                                Seq[Prompt],
                                                Supervised[Seq[GeneratedText]],
                                            ),
                                            output=Seq[GeneratedText],
                                            random_state=seed,
                                            registry=algorithm_registry,
                                            pop_size=1,
                                            evaluation_timeout=2 * Hour,
                                            memory_limit=30 * Gb,
                                            # multi-objective baseline uses 48 hours for search timeout
                                            search_timeout=24 * Hour,
                                            cross_validation_steps=1,
                                            stratified_cross_validation=False,
                                            # Objective functions. Multi-objective experiments use macro_f1_plain and evaluation_time
                                            objectives=objectives,
                                            # baseline uses original search algorithm
                                            search_algorithm=NSPEWarmStartSearch,
                                            # warm_start is None if baseline, otherwise it is the prepared WarmStart object
                                            warm_start=ws,
                                        )

                                        ml.fit(X_train, y_train)
                                    except ValueError as e:
                                        print(f"{e}. Generated: {config_name}")
                                    except Exception as e:
                                        print(
                                            f"Unexpected error during config '{config_name}': {e}"
                                        )
                                        return


if __name__ == "__main__":
    main()
