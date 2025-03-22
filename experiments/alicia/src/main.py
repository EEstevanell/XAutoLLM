#!/usr/bin/env python3
"""
IBERMAT Text Classification Experiment

This script implements a text classification experiment to detect whether text is human or machine generated
using the IBERMAT dataset. The script follows the AutoGOAL framework patterns but doesn't use warmstarting.

Usage:
    python main.py
"""

import os
import sys
import logging
from autogoal.search._nspge import NSPESearch
from autogoal_transformers._manual import FineTuneGenLLMClassifier, FineTuneLLMEmbeddingClassifier, LoraGenLLMClassifier, LoraLLMEmbeddingClassifier, PartialFineTuneGenLLMClassifier, PartialFineTuneLLMEmbeddingClassifier
import pandas as pd
import numpy as np
from pathlib import Path
import time
import random
from sklearn.model_selection import train_test_split
from autogoal.utils import Gb, Min, Hour

# Add parent directory to path to import common modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set fixed experiment parameters
EXPERIMENT_ID = f"ibermat_classification_{int(time.time())}"
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/IBERMAT.csv"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output"))
RANDOM_SEED = 42
TIME_BUDGET = 48*Hour
EVAL_TIMEOUT = 1.5*Hour
MEMORY_LIMIT = 8*Gb

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
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def set_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seeds set to {seed} for reproducibility")

class IBERMATDataset(Dataset):
    """
    Dataset class for IBERMAT text classification.
    """
    
    def __init__(self, data_path=DATA_PATH):
        """Initialize the IBERMAT dataset."""
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"IBERMAT dataset not found at {self.data_path}")
            
    def load(self):
        """Load the IBERMAT dataset and create train/test splits."""
        logger.info(f"Loading IBERMAT dataset from {self.data_path}")
        
        # Load CSV with semicolon separator
        df = pd.read_csv(self.data_path, sep=";", encoding="utf-8")
        
        logger.info(f"Dataset loaded with {len(df)} samples")
        logger.info(f"Column names: {df.columns.tolist()}")
        
        # Check if required columns exist
        required_columns = ["TEXT", "CLASS"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in dataset: {missing_columns}")
        
        # Extract texts and labels
        texts = df["TEXT"].tolist()
        labels = df["CLASS"].tolist()
        
        # Encode labels: HUMAN -> 0, MACHINE -> 1
        label_mapping = {"HUMAN": 0, "MACHINE": 1}
        y = np.array([label_mapping[label] for label in labels])
        
        # Print class distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_name = "HUMAN" if label == 0 else "MACHINE"
            logger.info(f"Class {label_name} ({label}): {count} samples ({count/len(y)*100:.2f}%)")
        
        # Create train/test split (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, y_train, X_test, y_test

def execute_experiment():
    """Execute the IBERMAT text classification experiment."""
    # Initialize environment
    start_time = time.time()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
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
    
    # Set threading environment variables for better performance
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    os.environ["NUMEXPR_NUM_THREADS"] = "8"
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
    os.environ["BLIS_NUM_THREADS"] = "8"
    
    # Load dataset
    dataset = IBERMATDataset()
    X_train, y_train, X_test, y_test = dataset.load()
    
    # Find appropriate algorithm classes for text classification
    algorithm_registry = (
        [
            FineTuneLLMEmbeddingClassifier,
            PartialFineTuneLLMEmbeddingClassifier,
            LoraLLMEmbeddingClassifier,
        ]
        + find_classes(include="WORD_EMB")
        + [
            FineTuneGenLLMClassifier,
            PartialFineTuneGenLLMClassifier,
            LoraGenLLMClassifier,
        ]
        + find_classes(include="TEXT_GEN")
    )
    
    logger.info(f"Found {len(algorithm_registry)} algorithm implementations for text classification")
    
    # Set up output paths
    output_dir = Path(OUTPUT_DIR)
    json_log_path = output_dir / f"{EXPERIMENT_ID}.json"
    results_path = output_dir / f"{EXPERIMENT_ID}_results.json"
    
    # Configure loggers
    loggers = [
        ConsoleLogger(),
        JsonLogger(str(json_log_path)),
    ]
    
    logger.info(f"Setting up experiment: {EXPERIMENT_ID}")
    logger.info(f"Results will be saved to: {json_log_path}")
    
    # Initialize AutoML
    automl = AutoML(
        input=(Seq[Sentence], Supervised[VectorDiscrete]),
        output=VectorDiscrete,
        registry=algorithm_registry,
        objectives=[macro_f1_plain],  # Using macro F1 score as primary metric
        observations=[
            ("Accuracy", accuracy),
            ("Evaluation Time", evaluation_time)
        ],
        maximize=(True,),  # Maximize F1 score
        search_algorithm=NSPESearch,  # Using Population-based Evolutionary Search
        search_timeout=TIME_BUDGET,
        random_state=RANDOM_SEED,
        memory_limit=MEMORY_LIMIT * Mb,
        evaluation_timeout=EVAL_TIMEOUT,
        cross_validation_steps=2,
    )
    
    # Run the experiment
    logger.info("Starting IBERMAT text classification experiment")
    try:
        automl.fit(X_train, y_train, logger=loggers)
        
        # Evaluate on test set
        logger.info("Evaluating best pipeline on test set")
        predictions = automl.predict(X_test)
        
        # Calculate metrics
        acc = accuracy(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='macro')
        prec = precision_score(y_test, predictions, average='macro')
        rec = recall_score(y_test, predictions, average='macro')
        
        # Log results
        logger.info("Test set results:")
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall: {rec:.4f}")
        
        # Save results to file
        results = {
            "experiment_id": EXPERIMENT_ID,
            "metrics": {
                "accuracy": float(acc),
                "f1_score": float(f1),
                "precision": float(prec),
                "recall": float(rec)
            },
            "runtime_seconds": time.time() - start_time,
            "parameters": {
                "random_seed": RANDOM_SEED,
                "time_budget": TIME_BUDGET,
                "eval_timeout": EVAL_TIMEOUT,
                "memory_limit": MEMORY_LIMIT,
                "cross_validation_steps": 3
            }
        }
        
        import json
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Experiment completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Results saved to {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during experiment execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    execute_experiment()