import os
from pathlib import Path
import sys
import time
from typing import Dict, List
from autogoal.meta_learning.distance.hybrid_distance import (
    HybridEuclideanCosineDistance,
)
from autogoal.meta_learning.feature_extraction.generative_task import (
    GenerativeTaskFeatureExtractor,
)
from autogoal.datasets import squad
from autogoal.meta_learning.warm_start import WarmStart
from autogoal.meta_learning.normalization import LogNormalizer, MinMaxNormalizer
from autogoal.ml import AutoML, evaluation_time
from autogoal.kb import Seq, Supervised, VectorDiscrete, Prompt, GeneratedText
from autogoal_transformers._generated import (
    TEXT_GEN_Gpt2,
)
from autogoal.search import JsonLogger, ConsoleLogger
from autogoal_transformers._manual import (
    FineTuneGenLLMTask,
    LoraGenLLMTask,
    PartialFineTuneGenLLMTask,
)
from autogoal.utils import Hour, Gb
from autogoal.search._warm_start_pge import NSPEWarmStartSearch
from autogoal.meta_learning._logging import ExperienceLogger
import logging
import nltk

def _format_squad_inputs(
    reference_texts: List[str], prediction_texts: List[str]
) -> Dict[str, List[Dict]]:
    """
    Formats lists of prediction and reference texts into the dictionary
    structure required by the Hugging Face SQuAD evaluate metric.

    Args:
        prediction_texts: A list of predicted answer strings.
        reference_texts: A list of corresponding ground-truth answer strings.

    Returns:
        A dictionary containing formatted 'predictions' and 'references' lists.

    Raises:
        ValueError: If the input lists have different lengths.
    """
    if len(prediction_texts) != len(reference_texts):
        raise ValueError("Prediction and reference lists must have the same length.")

    formatted_predictions = []
    formatted_references = []

    for i, (pred_text, ref_text) in enumerate(zip(prediction_texts, reference_texts)):
        # Generate a unique ID based on the index
        q_id = str(i)

        # Format prediction
        formatted_predictions.append(
            {"prediction_text": str(pred_text), "id": q_id}  # Ensure it's a string
        )

        # Format reference
        # The 'text' field must be a list of strings, even if there's only one answer [2, 3].
        formatted_references.append(
            {
                "answers": {
                    "text": [str(ref_text)],  # Ensure it's a string and wrap in a list
                    "answer_start": [],  # answer_start is often required but can be empty if only text is used
                },
                "id": q_id,
            }
        )

    return {"predictions": formatted_predictions, "references": formatted_references}


def compute_squad_f1(reference_texts: List[str], prediction_texts: List[str], *args, **kwargs) -> float:
    """
    Computes the official SQuAD F1 score given lists of prediction and reference texts.

    Args:
        prediction_texts: A list of predicted answer strings.
        reference_texts: A list of corresponding ground-truth answer strings.

    Returns:
        float: The average F1 score (0-100). Returns -1.0 if metric loading failed.
    """
    from evaluate import load

    squad_metric = load("squad")

    if squad_metric is None:
        print("SQuAD metric not loaded. Cannot compute F1 score.")
        return -1.0  # Indicate error

    try:
        formatted_data = _format_squad_inputs(prediction_texts, reference_texts)
        results = squad_metric.compute(
            predictions=formatted_data["predictions"],
            references=formatted_data["references"],
        )
        # The metric returns scores out of 100 [2, 3]
        print(f"F1 Score: {results['f1']}")
        # Return the F1 score
        return results["f1"]
    except ValueError as ve:
        print(f"Input Error: {ve}")
        return -1.0
    except Exception as e:
        print(f"Error during F1 computation: {e}")
        return -1.0


def compute_squad_exact_match(
    reference_texts: List[str], prediction_texts: List[str], *args, **kwargs
) -> float:
    """
    Computes the official SQuAD Exact Match (EM) score given lists of prediction and reference texts.

    Args:
        prediction_texts: A list of predicted answer strings.
        reference_texts: A list of corresponding ground-truth answer strings.

    Returns:
        float: The average Exact Match score (0-100). Returns -1.0 if metric loading failed.
    """
    from evaluate import load

    squad_metric = load("squad")

    if squad_metric is None:
        print("SQuAD metric not loaded. Cannot compute Exact Match score.")
        return -1.0  # Indicate error

    try:
        formatted_data = _format_squad_inputs(prediction_texts, reference_texts)
        results = squad_metric.compute(
            predictions=formatted_data["predictions"],
            references=formatted_data["references"],
        )
        # The metric returns scores out of 100 [2, 3]
        print(f"Exact Match Score: {results['exact_match']}")
        # Return the Exact Match score
        return results["exact_match"]
    except ValueError as ve:
        print(f"Input Error: {ve}")
        return -1.0
    except Exception as e:
        print(f"Error during Exact Match computation: {e}")
        return -1.0


# Import AutoGOAL components
try:
    # Import torch first to set random seeds
    import torch
    import numpy as np
    from autogoal.datasets import Dataset
    from autogoal.kb import Seq, Supervised, VectorDiscrete, Sentence
    from autogoal.ml import AutoML, evaluation_time, accuracy
    from autogoal.search import ConsoleLogger, JsonLogger
    from autogoal.search import PESearch
    from autogoal_contrib import find_classes
    from autogoal.datasets.semeval_2023_task_8_1 import macro_f1_plain
    from autogoal.utils import Hour, Mb
    from sklearn.metrics import f1_score, precision_score, recall_score
except ImportError as e:
    sys.exit(1)

# Set fixed experiment parameters
EXPERIMENT_ID = f"squad_{int(time.time())}"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output/"))
# Define path to squad.json at the workspace root
SQUAD_JSON_PATH = Path(__file__).resolve().parent / "squad.json"
RANDOM_SEED = 42
TIME_BUDGET = 48 * Hour
EVAL_TIMEOUT = 1.5 * Hour
MEMORY_LIMIT = 8 * Gb


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def set_seeds(seed=42):
    # Configure CUDA if available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(0)  # Use first GPU by default
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {device_name}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available. Using CPU.")
    pass


def main():
    nltk.download("all")

    # Set random seeds for reproducibility
    set_seeds()

    from autogoal.utils._process import initialize_cuda_multiprocessing

    initialize_cuda_multiprocessing()

    # Configure CUDA if available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(0)  # Use first GPU by default
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {device_name}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available. Using CPU.")

    # Load dataset
    X_train, y_train, X_test, y_test = squad.load(True)
    X_train = X_train[:1000]  # Limit to 1000 samples for testing
    y_train = y_train[:1000]  # Limit to 1000 samples for testing
    X_test = X_test[:1000]  # Limit to 1000 samples for testing
    y_test = y_test[:1000]  # Limit to 1000 samples for testing

    # Find appropriate algorithm classes for text classification
    algorithm_registry = (
        [
            # FineTuneGenLLMTask,
            PartialFineTuneGenLLMTask,
            LoraGenLLMTask,
            TEXT_GEN_Gpt2,
        ]
        # + find_classes(include="TEXT_GEN")
    )

    output_dir = Path(OUTPUT_DIR)
    json_log_path = output_dir / f"llms-{EXPERIMENT_ID}.json"
    results_path = output_dir / f"llms-{EXPERIMENT_ID}_results.json"

    logger.info(
        f"Found {len(algorithm_registry)} algorithm implementations for question answering"
    )

    # Set up output paths
    output_dir = Path(OUTPUT_DIR)
    json_log_path = output_dir / f"llms-{EXPERIMENT_ID}.json"
    results_path = output_dir / f"llms-{EXPERIMENT_ID}_results.json"

    # Configure loggers
    loggers = [
        ConsoleLogger(),
        JsonLogger(str(json_log_path)),
    ]

    logger.info(f"Setting up experiment: {EXPERIMENT_ID}")
    logger.info(f"Results will be saved to: {json_log_path}")

    config = dict()
    warm_start = WarmStart(
        dataset_feature_extractor=GenerativeTaskFeatureExtractor,
        positive_min_threshold=config.get("positive_min_threshold", 0.2),
        adaptative_negative_alpha_limit=config.get("adaptative_negative_alpha_limit", None),
        k_pos=config.get("k_pos", 10000),
        k_neg=config.get("k_neg", 10000),
        distance=config.get("distance", HybridEuclideanCosineDistance),
        normalizers=config.get("normalizers", [LogNormalizer(), MinMaxNormalizer()]),
        exclude=f"{config.get('exclude', '')}|_warmstart|seed",
        include=config.get("include", None),
        beta_scale=config.get("beta_scale", 1.0),
        beta=config.get("beta", None),
        max_alpha=config.get("max_alpha", 0.05),
        min_alpha=config.get("min_alpha", -0.02),
        utility_function=config.get("utility_function", "linear_front"),
        metrics=["exact_match", "f1_score"],  # <-- Add this line
    )

    import json

    # try loading the dataset features from <dataset_name>.json
    current_dataset_features = None

    try:
        with open(SQUAD_JSON_PATH, "r") as f:  # Use SQUAD_JSON_PATH
            current_dataset_features = json.load(f)
    except FileNotFoundError:  # Be more specific with the exception
        logger.info(f"Cache file {SQUAD_JSON_PATH} not found. Will compute features.")
    except Exception as e:  # Catch other potential errors
        logger.warning(
            f"Error loading cache file {SQUAD_JSON_PATH}: {e}. Will recompute features."
        )

    # initialize the WarmStart object with the current task features
    warm_start.pre_warm_up(
        X_train, y_train, current_dataset_meta_features=current_dataset_features
    )

    # Create or update <dataset_name>.json with the current task features (cache)
    try:  # Add try-except for writing
        with open(SQUAD_JSON_PATH, "w") as f:  # Use SQUAD_JSON_PATH
            json.dump(warm_start.current_dataset_features, f, indent=2)
        logger.info(f"Successfully cached features to {SQUAD_JSON_PATH}")
    except Exception as e:
        logger.error(f"Error caching features to {SQUAD_JSON_PATH}: {e}")

    objectives = [
        {
            "name": "exact_match",
            "metric": compute_squad_exact_match,
            "maximize": True,
        },
        {
            "name": "f1_score",
            "metric": compute_squad_f1,
            "maximize": True,
        },
    ]

    optimizer = NSPEWarmStartSearch
    seed = 42

    algorithm_registry = (
        [
            FineTuneGenLLMTask,
            LoraGenLLMTask,
            PartialFineTuneGenLLMTask,
            TEXT_GEN_Gpt2,
        ]
        # + find_classes(include="TEXT_GEN")
    )

    model = AutoML(
        input=(Seq[Prompt], Supervised[Seq[GeneratedText]]),
        output=Seq[GeneratedText],
        random_state=seed,
        registry=algorithm_registry,
        evaluation_timeout=1.5 * Hour,
        memory_limit=35 * Gb,
        # multi-objective baseline uses 48 hours for search timeout
        search_timeout=5 * Hour,
        cross_validation_steps=1,
        stratified_cross_validation=True,
        # Objective functions. Multi-objective experiments use macro_f1_plain and evaluation_time
        objectives=objectives,
        # Additional observations for logging
        observations=[("Evaluation Time", evaluation_time)],
        # baseline uses original search algorithm
        search_algorithm=optimizer,
        # warm_start is None if baseline, otherwise it is the prepared WarmStart object
        warm_start=warm_start,
    )

    # Initialize loggers
    loggers = [
        ConsoleLogger(),
        ExperienceLogger(
            dataset_features=warm_start.current_dataset_features,
            system_features=warm_start.current_system_features,
            dataset_feature_extractor_name="GenerativeTaskFeatureExtractor",
            system_feature_extractor_name="SystemFeatureExtractor",
            alias="squad",
        ),
    ]

    # Run the experiment
    logger.info("Starting SQuAD experiment")
    try:
        start_time = time.time()
        model.fit(X_train, y_train, logger=loggers)

        # Evaluate on test set
        logger.info("Evaluating best pipeline on test set")
        predictions = model.predict(X_test)

        # Calculate metrics
        em = compute_squad_exact_match(y_test, predictions)
        f1 = compute_squad_f1(y_test, predictions)

        # Log results
        logger.info("Test set results:")
        logger.info(f"Exact Match: {em:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")

        # Save results to file
        results_obj = {
            "experiment_id": EXPERIMENT_ID,
            "metrics": {
                "exact_match": float(em),
                "f1_score": float(f1),
            },
            "runtime_seconds": time.time() - start_time,
            "parameters": {
                "random_seed": RANDOM_SEED,
                "time_budget": TIME_BUDGET,
                "eval_timeout": EVAL_TIMEOUT,
                "memory_limit": MEMORY_LIMIT,
                "cross_validation_steps": 1,
            },
        }
        import json

        with open(results_path, "w") as f:
            json.dump(results_obj, f, indent=2)
        logger.info(f"Experiment completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Results saved to {results_path}")
        return results_obj

    except Exception as e:
        logger.error(f"Error during experiment execution: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    main()
