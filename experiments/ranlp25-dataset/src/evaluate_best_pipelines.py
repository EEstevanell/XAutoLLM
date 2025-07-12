#!/usr/bin/env python3
"""Evaluate the best encoder and traditional pipelines on the RANLP25 dataset."""

import argparse
import json
import logging
import random
import sys
from pathlib import Path


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RANDOM_SEED = 42
DATA_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = DATA_DIR / "output"


class RANLP25Dataset:
    """Utility to load the RANLP25 train/test splits."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.train_path = data_dir / "train.csv"
        self.test_path = data_dir / "test.csv"
        if not self.train_path.exists() or not self.test_path.exists():
            raise FileNotFoundError("RANLP25 dataset files not found")

    def load(self):
        import pandas as pd

        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        train_df.fillna("", inplace=True)
        test_df.fillna("", inplace=True)

        X_train = (train_df["title"] + ". " + train_df["abstract"]).tolist()
        X_test = (test_df["title"] + ". " + test_df["abstract"]).tolist()

        all_labels = pd.Categorical(train_df["type"]).categories
        label_mapping = {lbl: idx for idx, lbl in enumerate(all_labels)}
        y_train = [label_mapping[t] for t in train_df["type"]]
        y_test = [label_mapping[t] for t in test_df["type"]]

        self.test_ids = test_df["paper_id"].tolist()
        self.label_mapping = label_mapping

        return X_train, y_train, X_test, y_test


def set_seeds(seed: int = RANDOM_SEED):
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_encoder_pipeline():
    """Return the best encoder pipeline defined in the experiment."""

    from autogoal.kb import Pipeline, Seq, Sentence, Supervised, Tensor, Discrete, Dense
    from autogoal_transformers.autogoal_transformers._manual import LoraLLMEmbeddingClassifier
    from autogoal_transformers.autogoal_transformers._generated import (
        WORD_EMB_Distilbert_Base_Multilingual_Cased,
    )

    return Pipeline(
        algorithms=[
            LoraLLMEmbeddingClassifier(
                inner_model=WORD_EMB_Distilbert_Base_Multilingual_Cased(),
                lora_r=1,
                lora_alpha=32,
                lora_dropout=0.2,
                lora_bias="lora_only",
                batch_size=8,
                max_length=700,
                learning_rate=1e-05,
                epochs=10,
                warmup_steps=1500,
                weight_decay=0.005,
                optimizer="adamw",
                gradient_accumulation_steps=2,
                lr_scheduler="linear",
                model_save="classifier",
                init_lora_weights="gaussian",
                fan_in_fan_out=False,
                early_stopping_delta=0.01,
                use_mixed_precision=True,
                use_gradient_clipping=True,
                gradient_clipping_max_norm=0.5,
                class_weighted_loss=False,
                num_workers="default",
            )
        ],
        input_types=(Seq[Sentence], Supervised[Tensor[1, Discrete, Dense]]),
    )


def get_traditional_pipeline():
    """Return the best traditional ML pipeline defined in the experiment."""

    from autogoal.kb import Pipeline, Seq, Sentence, Supervised, Tensor, Categorical, Dense
    from autogoal_sklearn import HashingVectorizer, PassiveAggressiveClassifier

    return Pipeline(
        algorithms=[
            HashingVectorizer(
                alternate_sign=False,
                binary=True,
                lowercase=True,
                n_features=2097151,
                norm="l1",
            ),
            PassiveAggressiveClassifier(
                C=9.991,
                average=False,
                early_stopping=False,
                fit_intercept=False,
                n_iter_no_change=3,
                shuffle=True,
                tol=0.001,
                validation_fraction=0.993,
            ),
        ],
        input_types=(Seq[Sentence], Supervised[Tensor[1, Categorical, Dense]]),
    )


def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds, average="macro"),
        "precision": precision_score(y_test, preds, average="macro"),
        "recall": recall_score(y_test, preds, average="macro"),
    }
    return metrics, preds


def main(output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seeds()

    dataset = RANLP25Dataset()
    X_train, y_train_int, X_test, y_test_int = dataset.load()

    # Labels as strings are required for scikit-learn wrappers
    y_train_str = [str(y) for y in y_train_int]
    y_test_str = [str(y) for y in y_test_int]

    label_rev = {v: k for k, v in dataset.label_mapping.items()}

    results_rows = []

    logger.info("Evaluating encoder pipeline")
    enc_pipeline = get_encoder_pipeline()
    enc_metrics, enc_preds = evaluate_pipeline(enc_pipeline, X_train, y_train_int, X_test, y_test_int)
    results_rows.append({"model": "encoders", **enc_metrics})
    enc_labels = [label_rev[int(p)] for p in enc_preds]
    enc_df = pd.DataFrame({"paper_id": dataset.test_ids, "prediction": enc_labels})
    enc_df.to_csv(output_dir / "encoders_best_predictions.csv", index=False)

    logger.info("Evaluating traditional pipeline")
    trad_pipeline = get_traditional_pipeline()
    trad_metrics, trad_preds = evaluate_pipeline(trad_pipeline, X_train, y_train_str, X_test, y_test_str)
    results_rows.append({"model": "traditional", **trad_metrics})
    trad_labels = [label_rev[int(p)] for p in trad_preds]
    trad_df = pd.DataFrame({"paper_id": dataset.test_ids, "prediction": trad_labels})
    trad_df.to_csv(output_dir / "traditional_best_predictions.csv", index=False)

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(output_dir / "best_models_results.csv", index=False)
    logger.info(f"Results saved to {output_dir / 'best_models_results.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where results will be written",
    )
    args = parser.parse_args()

    # Ensure contrib packages are discoverable when the script actually runs
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.append(str(REPO_ROOT / "autogoal"))
    sys.path.append(str(REPO_ROOT / "autogoal-contrib"))

    main(args.output_dir)
