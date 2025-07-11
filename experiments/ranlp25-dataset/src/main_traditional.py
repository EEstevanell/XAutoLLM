#!/usr/bin/env python3
"""
Text classification experiment for the RANLP25 dataset using traditional ML models.
The script follows the structure of other experiments in this repository.
"""
import os
import sys
import logging
from pathlib import Path
import time
import random

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from autogoal.kb import Seq, Supervised, VectorCategorical, Sentence
from autogoal.ml import AutoML, evaluation_time, accuracy
from autogoal.search import NSPESearch, ConsoleLogger, JsonLogger
from autogoal_contrib import find_classes
from autogoal.datasets.semeval_2023_task_8_1 import macro_f1_plain
from autogoal.utils import Gb, Hour, Mb, Min
from autogoal.utils._objective import Objective

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

EXPERIMENT_ID = f"ranlp25_traditional_{int(time.time())}"
DATA_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = DATA_DIR / "output"
RANDOM_SEED = 42
TIME_BUDGET = 24 * Hour
EVAL_TIMEOUT = 30 * Min
MEMORY_LIMIT = 8 * Gb

class RANLP25Dataset:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.train_path = data_dir / "train.csv"
        self.test_path = data_dir / "test.csv"
        if not self.train_path.exists() or not self.test_path.exists():
            raise FileNotFoundError("RANLP25 dataset files not found")

    def load(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        train_df.fillna("", inplace=True)
        test_df.fillna("", inplace=True)

        X_train = (train_df["title"] + ". " + train_df["abstract"]).tolist()
        X_test = (test_df["title"] + ". " + test_df["abstract"]).tolist()

        all_labels = pd.Categorical(train_df["type"]).categories
        label_mapping = {lbl: idx for idx, lbl in enumerate(all_labels)}
        y_train = [str(label_mapping[t]) for t in train_df["type"]]
        y_test = [str(label_mapping[t]) for t in test_df["type"]]

        return X_train, y_train, X_test, y_test

def set_seeds(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def execute_experiment():
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seeds()

    dataset = RANLP25Dataset()
    X_train, y_train, X_test, y_test = dataset.load()

    import autogoal_nltk, autogoal_sklearn
    algorithm_registry = find_classes(modules=[autogoal_nltk, autogoal_sklearn])
    logger.info(f"Found {len(algorithm_registry)} algorithms")

    json_log_path = OUTPUT_DIR / f"{EXPERIMENT_ID}.json"
    results_path = OUTPUT_DIR / f"{EXPERIMENT_ID}_results.json"
    preds_path = OUTPUT_DIR / f"{EXPERIMENT_ID}_test_predictions.json"
    
    automl = AutoML(
        input=(Seq[Sentence], Supervised[VectorCategorical]),
        output=VectorCategorical,
        registry=algorithm_registry,
        objectives=Objective(name="f1", metric=macro_f1_plain, maximize=True),
        observations=[("Accuracy", accuracy), ("Evaluation Time", evaluation_time)],
        maximize=True,
        search_algorithm=NSPESearch,
        search_timeout=TIME_BUDGET,
        random_state=RANDOM_SEED,
        memory_limit=MEMORY_LIMIT * Mb,
        evaluation_timeout=EVAL_TIMEOUT,
        cross_validation_steps=5,
    )

    loggers = [ConsoleLogger(), JsonLogger(str(json_log_path))]

    try:
        automl.fit(X_train, y_train, logger=loggers)
        predictions = automl.predict(X_test)
        acc = accuracy(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="macro")
        prec = precision_score(y_test, predictions, average="macro")
        rec = recall_score(y_test, predictions, average="macro")

        results = {
            "experiment_id": EXPERIMENT_ID,
            "metrics": {
                "accuracy": float(acc),
                "f1_score": float(f1),
                "precision": float(prec),
                "recall": float(rec),
            },
            "runtime_seconds": time.time() - start,
        }
        with open(results_path, "w") as f:
            import json
            json.dump(results, f, indent=2)
        with open(preds_path, "w") as f:
            import json
            json.dump([int(p) for p in predictions], f, indent=2)
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Predictions saved to {preds_path}")
    except Exception as e:
        logger.error(f"Experiment error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    execute_experiment()
