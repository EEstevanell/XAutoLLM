import os
import sys
import logging
from typing import Dict, List
from autogoal.kb._semantics import ContextQuestionPair, GeneratedText, Prompt
from autogoal.search._nspge import NSPESearch
from autogoal_transformers._manual import FineTuneGenLLMTask, PartialFineTuneGenLLMTask, LoraGenLLMTask
from autogoal_transformers._generated import TEXT_GEN_Microsoft_Phi_4_Mini_Reasoning, TEXT_GEN_Mistralai_Mistral_Nemo_Instruct_Fp8_2407, TEXT_GEN_Deepseek_Ai_Deepseek_R1_Distill_Qwen_7B
import pandas as pd
import numpy as np
from pathlib import Path
import time
import random
from sklearn.model_selection import train_test_split
from autogoal.utils import Gb, Min, Hour
import autogoal.datasets.squad as squad 

def _format_squad_inputs(reference_texts: List[str], prediction_texts: List[str]) -> Dict[str, List[Dict]]:
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
        formatted_predictions.append({
            'prediction_text': str(pred_text), # Ensure it's a string
            'id': q_id
        })

        # Format reference
        # The 'text' field must be a list of strings, even if there's only one answer [2, 3].
        formatted_references.append({
            'answers': {
                'text': [str(ref_text)], # Ensure it's a string and wrap in a list
                'answer_start': [] # answer_start is often required but can be empty if only text is used
                },
            'id': q_id
        })

    return {"predictions": formatted_predictions, "references": formatted_references}

def compute_squad_f1(reference_texts: List[str], prediction_texts: List[str]) -> float:
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
        return -1.0 # Indicate error

    try:
        formatted_data = _format_squad_inputs(prediction_texts, reference_texts)
        results = squad_metric.compute(
            predictions=formatted_data["predictions"],
            references=formatted_data["references"]
        )
        # The metric returns scores out of 100 [2, 3]
        print(f"F1 Score: {results['f1']}")
        # Return the F1 score
        return results['f1']
    except ValueError as ve:
        print(f"Input Error: {ve}")
        return -1.0
    except Exception as e:
        print(f"Error during F1 computation: {e}")
        return -1.0

def compute_squad_exact_match(reference_texts: List[str], prediction_texts: List[str]) -> float:
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
        return -1.0 # Indicate error

    try:
        formatted_data = _format_squad_inputs(prediction_texts, reference_texts)
        results = squad_metric.compute(
            predictions=formatted_data["predictions"],
            references=formatted_data["references"]
        )
        # The metric returns scores out of 100 [2, 3]
        print(f"Exact Match Score: {results['exact_match']}")
        # Return the Exact Match score
        return results['exact_match']
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
RANDOM_SEED = 42
TIME_BUDGET = 48*Hour
EVAL_TIMEOUT = 1.5*Hour
MEMORY_LIMIT = 8*Gb



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
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
    X_train, y_train, X_test, y_test = squad.load()

    # We must convert from tuple(context,question) to a single string for the instruction.
    # The actual X_train and X_test should be lists of strings.
    X_train = [f"Context: {context} Question: {question} Answer:" for context, question in X_train]
    X_test = [f"Context: {context} Question: {question} Answer:" for context, question in X_test]


    # Find appropriate algorithm classes for text classification
    algorithm_registry = (
        [
            # FineTuneGenLLMTask,
            PartialFineTuneGenLLMTask,
            LoraGenLLMTask,
            TEXT_GEN_Microsoft_Phi_4_Mini_Reasoning
        ]
        # + find_classes(include="TEXT_GEN")
    )

    output_dir = Path(OUTPUT_DIR)
    json_log_path = output_dir / f"llms-{EXPERIMENT_ID}.json"
    results_path = output_dir / f"llms-{EXPERIMENT_ID}_results.json"

    logger.info(f"Found {len(algorithm_registry)} algorithm implementations for question answering")
    
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
    
    # Initialize AutoML
    automl = AutoML(
        input=(Seq[Prompt], Supervised[Seq[GeneratedText]]),
        output=Seq[GeneratedText],
        registry=algorithm_registry,
        objectives=(compute_squad_exact_match, compute_squad_f1),  # Using Exact Match and F1 score as primary metrics
        observations=[
            ("Evaluation Time", evaluation_time)
        ],
        maximize=True,  # Maximize F1 score
        search_algorithm=NSPESearch,  # Using Population-based Evolutionary Search
        search_timeout=TIME_BUDGET,
        random_state=RANDOM_SEED,
        memory_limit=MEMORY_LIMIT * Mb,
        evaluation_timeout=EVAL_TIMEOUT,
        cross_validation_steps=1,
    )
    
    # Run the experiment
    logger.info("Starting SQuAD experiment")
    try:
        start_time = time.time()
        automl.fit(X_train, y_train, logger=loggers)
        
        # Evaluate on test set
        logger.info("Evaluating best pipeline on test set")
        predictions = automl.predict(X_test)
        
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
                "cross_validation_steps": 1
            }
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