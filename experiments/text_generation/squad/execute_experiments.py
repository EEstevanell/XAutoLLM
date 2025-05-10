import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional
from autogoal.meta_learning.distance import (
    CosineDistance,
)
from autogoal.meta_learning.feature_extraction.generative_task import (
    GenerativeTaskFeatureExtractor,
)
from autogoal.datasets import squad, cnn_dailymail, drop
from autogoal.meta_learning.warm_start import WarmStart
from autogoal.meta_learning.normalization import LogNormalizer, MinMaxNormalizer
from autogoal.ml import AutoML, evaluation_time
from autogoal.kb import Seq, Supervised, Prompt, GeneratedText
from autogoal_transformers._generated import (
    TEXT_GEN_Gpt2,
    TEXT_GEN_Meta_Llama_Llama_32_1B,
    TEXT_GEN_Microsoft_Phi_4_Mini_Instruct,
    TEXT_GEN_Microsoft_Phi_35_Mini_Instruct,
    TEXT_GEN_Mistralai_Mistral_7B_V01,
    TEXT_GEN_Facebook_Bart_Base,
    TEXT_GEN_Deepseek_Ai_Deepseek_R1_Distill_Qwen_7B,
    TEXT_GEN_Google_T5_T5_Small
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
import numpy as np

def _compute_rouge_scores(reference_texts: List[str], prediction_texts: List[str]) -> Dict[str, float]:
    """
    Helper function to compute all ROUGE scores using the Hugging Face evaluate library.

    Args:
        reference_texts: A list of reference summary strings.
        prediction_texts: A list of corresponding predicted summary strings.

    Returns:
        A dictionary containing ROUGE scores (e.g., rouge1, rouge2, rougeL, rougeLsum).
        Returns an empty dictionary if metric loading or computation fails.
    """
    if len(prediction_texts) != len(reference_texts):
        logger.error("Prediction and reference lists must have the same length.")
        raise ValueError("Prediction and reference lists must have the same length.")

    # Handle empty lists: ROUGE scores are not well-defined or are zero.
    if len(prediction_texts) == 0 or len(reference_texts) == 0:
        logger.error(
            "ROUGE computation: One or both input lists (predictions, references) are empty. Returning empty scores."
        )
        return {}

    try:
        from evaluate import load
        rouge_metric = load("rouge")
    except Exception as e:
        logger.error(f"Failed to load ROUGE metric from Hugging Face evaluate: {e}")
        return {}

    if rouge_metric is None:
        logger.error("ROUGE metric not loaded successfully.")
        return {}

    try:
        results = rouge_metric.compute(
            predictions=prediction_texts,
            references=reference_texts,
            # Optional: use_stemmer=True can be added for Porter stemmer application
        )
        # The results are typically like:
        # {'rouge1': 0.45, 'rouge2': 0.25, 'rougeL': 0.40, 'rougeLsum': 0.42}
        # These are F-measure scores by default for rouge1, rouge2, rougeL.
        # For rougeLsum, it's also an F-measure.
        return results
    except ValueError as ve:
        logger.error(f"Input error during ROUGE computation: {ve}")
        return {}
    except Exception as e:
        logger.error(f"Error during ROUGE computation: {e}")
        return {}

def compute_rouge1(reference_texts: List[str], prediction_texts: List[str], *args, **kwargs) -> float:
    """
    Computes the ROUGE-1 score.

    Args:
        reference_texts: A list of reference summary strings.
        prediction_texts: A list of corresponding predicted summary strings.

    Returns:
        float: The ROUGE-1 score (0-1). Returns -1.0 on error.
    """
    results = _compute_rouge_scores(reference_texts, prediction_texts)
    if results and "rouge1" in results:
        logger.info(f"ROUGE-1 Score: {results['rouge1']}")
        return results["rouge1"]
    return -1.0

def compute_rouge2(reference_texts: List[str], prediction_texts: List[str], *args, **kwargs) -> float:
    """
    Computes the ROUGE-2 score.

    Args:
        reference_texts: A list of reference summary strings.
        prediction_texts: A list of corresponding predicted summary strings.

    Returns:
        float: The ROUGE-2 score (0-1). Returns -1.0 on error.
    """
    results = _compute_rouge_scores(reference_texts, prediction_texts)
    if results and "rouge2" in results:
        logger.info(f"ROUGE-2 Score: {results['rouge2']}")
        return results["rouge2"]
    return -1.0

def compute_rougeL(reference_texts: List[str], prediction_texts: List[str], *args, **kwargs) -> float:
    """
    Computes the ROUGE-L score (Longest Common Subsequence at sentence level).

    Args:
        reference_texts: A list of reference summary strings.
        prediction_texts: A list of corresponding predicted summary strings.

    Returns:
        float: The ROUGE-L score (0-1). Returns -1.0 on error.
    """
    results = _compute_rouge_scores(reference_texts, prediction_texts)
    if results and "rougeL" in results:
        logger.info(f"ROUGE-L Score: {results['rougeL']}")
        return results["rougeL"]
    return -1.0

def compute_rougeLsum(reference_texts: List[str], prediction_texts: List[str], *args, **kwargs) -> float:
    """
    Computes the ROUGE-Lsum score (Longest Common Subsequence at summary level).

    Args:
        reference_texts: A list of reference summary strings.
        prediction_texts: A list of corresponding predicted summary strings.

    Returns:
        float: The ROUGE-Lsum score (0-1). Returns -1.0 on error.
    """
    results = _compute_rouge_scores(reference_texts, prediction_texts)
    if results and "rougeLsum" in results:
        logger.info(f"ROUGE-Lsum Score: {results['rougeLsum']}")
        return results["rougeLsum"]
    return -1.0

# Import AutoGOAL components
try:
    # Import torch first to set random seeds
    import torch
    import numpy as np
    from autogoal.kb import Seq, Supervised
    from autogoal.ml import AutoML, evaluation_time
    from autogoal.search import ConsoleLogger, JsonLogger
    from autogoal_contrib import find_classes
    from autogoal.utils import Hour, Mb
    
    # --- Limit GPU VRAM usage to 14GB (best practice, before torch import) ---
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the main GPU

    try:
        import torch
        total_mem = torch.cuda.get_device_properties(0).total_memory
        limit_bytes = 14 * 1024 ** 3
        fraction = min(1.0, limit_bytes / total_mem)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        print(f"[INFO] Set per-process GPU memory fraction to {fraction:.3f} (max 14GB)")
    except Exception as e:
        print(f"[WARN] Could not set per-process memory fraction: {e}")
        
except ImportError as e:
    sys.exit(1)

# Set fixed experiment parameters
EXPERIMENT_ID = f"squad_{int(time.time())}"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output/"))
# Define path to squad.json at the workspace root
SQUAD_JSON_PATH = Path(__file__).resolve().parent / "squad.json"
CNN_DAILYMAIL_JSON_PATH = Path(__file__).resolve().parent / "drop.json"
RANDOM_SEED = 42
TIME_BUDGET = 48 * Hour
EVAL_TIMEOUT = 1.5 * Hour
MEMORY_LIMIT = 8 * Gb

# Configure logging properly
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)


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

def _serialize_task_features(task_features: Dict[str, Optional[np.ndarray]]) -> Dict[str, Optional[List[float]]]:
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

def _deserialize_task_features(loaded_features: Dict[str, Optional[List[float]]]) -> Dict[str, Optional[np.ndarray]]:
    """Converts list features in a dictionary loaded from JSON back to np.ndarray."""
    if loaded_features is None:
        return None
    deserialized = {}
    for key, value in loaded_features.items():
        if isinstance(value, list):
            deserialized[key] = np.array(value, dtype=float) # Assuming float for features
        elif value is None:
            deserialized[key] = None
        else:
            # Should ideally not happen
            deserialized[key] = np.array(value, dtype=float) if isinstance(value, (list, tuple)) else value
    return deserialized

def main():
    # Disable tokenizers parallelism to avoid warnings after forking
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    X_train, y_train, X_test, y_test = drop.load(True)
    X_train = X_train  # Limit to 1000 samples for testing
    y_train = y_train  # Limit to 1000 samples for testing
    X_test = X_test  # Limit to 1000 samples for testing
    y_test = y_test # Limit to 1000 samples for testing
    
    # X_train, y_train, X_test, y_test = squad.load(True)
    # X_train = X_train  # Limit to 1000 samples for testing
    # y_train = y_train  # Limit to 1000 samples for testing
    # X_test = X_test  # Limit to 1000 samples for testing
    # y_test = y_test # Limit to 1000 samples for testing


    # Find appropriate algorithm classes for text classification
    algorithm_registry = (
        [
            # FineTuneGenLLMTask,
            PartialFineTuneGenLLMTask, 
            # LoraGenLLMTask,
            # TEXT_GEN_Gpt2,
            TEXT_GEN_Meta_Llama_Llama_32_1B,
            # TEXT_GEN_Microsoft_Phi_4_Mini_Instruct,
            # TEXT_GEN_Microsoft_Phi_35_Mini_Instruct,
            # TEXT_GEN_Mistralai_Mistral_7B_V01,
            # TEXT_GEN_Facebook_Bart_Base,
            # TEXT_GEN_Deepseek_Ai_Deepseek_R1_Distill_Qwen_7B,
            # TEXT_GEN_Google_T5_T5_Small
        ]
        # + find_classes(include="TEXT_GEN", exclude="T5")
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

    # Construct exclude pattern carefully
    exclude_terms = ["_warmstart", "seed"]
    user_exclude = config.get('exclude') # Get user exclude, defaults to None if not present
    if user_exclude:
        exclude_terms.insert(0, user_exclude)
    
    # Join terms with | and group them to ensure they are treated as whole patterns
    # Only create the pattern if there are terms to exclude
    exclude_pattern = "|".join(f"({term})" for term in exclude_terms) if exclude_terms else None

    warm_start = WarmStart(
        dataset_feature_extractor=GenerativeTaskFeatureExtractor,
        positive_min_threshold=config.get("positive_min_threshold", 0.2),
        adaptative_negative_alpha_limit=config.get("adaptative_negative_alpha_limit", None),
        k_pos=config.get("k_pos", 10000),
        k_neg=config.get("k_neg", 10000),
        distance=config.get("distance", CosineDistance),
        normalizers=config.get("normalizers", [LogNormalizer(), MinMaxNormalizer()]),
        exclude=exclude_pattern, # Use the carefully constructed pattern
        include=config.get("include", None),
        beta_scale=config.get("beta_scale", 1.0),
        beta=config.get("beta", None),
        max_alpha=config.get("max_alpha", 0.05),
        min_alpha=config.get("min_alpha", -0.02),
        utility_function=config.get("utility_function", "linear_front"),
        metrics=["evaluation_time", "rouge-L"],  # <-- Add this line
    )

    import json

    # try loading the dataset features from <dataset_name>.json
    current_task_features = None

    try:
        with open(CNN_DAILYMAIL_JSON_PATH, "r") as f:  # Use CNN_DAILYMAIL_JSON_PATH
            loaded_f = json.load(f)
            current_task_features = _deserialize_task_features(loaded_f) # Deserialize here
    except FileNotFoundError:  # Be more specific with the exception
        logger.info(f"Cache file {CNN_DAILYMAIL_JSON_PATH} not found. Will compute features.")
    except Exception as e:  # Catch other potential errors
        logger.warning(
            f"Error loading cache file {CNN_DAILYMAIL_JSON_PATH}: {e}. Will recompute features."
        )

    # initialize the WarmStart object with the current task features
    warm_start.pre_warm_up(
        X_train, y_train, current_task_features=current_task_features
    )
    
    # Create or update <dataset_name>.json with the current task features (cache)
    try:  # Add try-except for writing
        with open(CNN_DAILYMAIL_JSON_PATH, "w") as f:  # Use SQUAD_JSON_PATH
            serialized_features = _serialize_task_features(warm_start.current_task_features) # Serialize here
            json.dump(serialized_features, f, indent=2)
        logger.info(f"Successfully cached features to {CNN_DAILYMAIL_JSON_PATH}")
    except Exception as e:
        logger.error(f"Error caching features to {CNN_DAILYMAIL_JSON_PATH}: {e}")

    objectives = [
        {
            "name": "f1",
            "metric": drop.compute_f1,
            "maximize": True,
        },
        {
            "name": "evaluation_time",
            "metric": evaluation_time,
            "maximize": False,
        },
    ]

    optimizer = NSPEWarmStartSearch
    seed = 42

    algorithm_registry = (
        [
            FineTuneGenLLMTask,
            LoraGenLLMTask,
            PartialFineTuneGenLLMTask,
        ]
        + find_classes(include="TEXT_GEN")
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
        stratified_cross_validation=False,
        # Objective functions. Multi-objective experiments use macro_f1_plain and evaluation_time
        objectives=objectives,
        # Additional observations for logging
        observations=[("exact match", drop.compute_exact_match)],
        # baseline uses original search algorithm
        search_algorithm=optimizer,
        # warm_start is None if baseline, otherwise it is the prepared WarmStart object
        warm_start=warm_start,
    )

    # Initialize loggers
    loggers = [
        ConsoleLogger(),
        ExperienceLogger(
            dataset_features=warm_start.current_task_features,
            system_features=warm_start.current_system_features,
            dataset_feature_extractor_name="GenerativeTaskFeatureExtractor",
            system_feature_extractor_name="SystemFeatureExtractor",
            alias="cnn_dailymail",
        ),
    ]

    # Run the experiment
    logger.info("Starting CNN/Dailymail experiment")
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
