from collections import Counter
import csv
import json
import re
import string
from typing import Dict, Tuple, List, Optional, Union

from autogoal.datasets import download, datapath
import evaluate

# --- DROP Official Metric Implementation (AllenNLP style, multi-span, multi-reference) ---

def compute_drop_f1_em_metric(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes the F1 and Exact Match (EM) scores for the DROP dataset using the Hugging Face evaluate library.
    The 'squad' metric is used as it handles extractive QA tasks with multiple references/predictions.

    Args:
        predictions (List[str]): Each is a JSON string decoding to a list of predicted answer spans.
                                 Example: '["answer span 1", "another span"]'
        references (List[str]): Each is a JSON string decoding to an object with "spans" (list of gold answer spans).
                                Example: '{"spans": ["gold span 1", "gold span 2"]}'
    Returns:
        Dict[str, float]: {"f1": ..., "exact_match": ...}
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    squad_metric = evaluate.load("squad")
    
    formatted_predictions = []
    formatted_references = []

    for i, (pred_json_str, ref_json_str) in enumerate(zip(predictions, references)):
        # Parse prediction
        try:
            pred_spans_list = json.loads(pred_json_str)
            if not isinstance(pred_spans_list, list): # Handle single string prediction
                pred_spans_list = [str(pred_spans_list)]
            prediction_text_for_squad = " ".join(pred_spans_list) if pred_spans_list else ""
        except Exception:
            prediction_text_for_squad = ""

        # Parse reference
        try:
            ref_obj = json.loads(ref_json_str)
            gold_spans = ref_obj.get("spans", [])
            if not isinstance(gold_spans, list):
                gold_spans = [str(gold_spans)]
            gold_spans = [str(s) for s in gold_spans]
        except Exception:
            gold_spans = [""]

        formatted_predictions.append({
            "id": str(i),
            "prediction_text": prediction_text_for_squad
        })
        formatted_references.append({
            "id": str(i),
            "answers": {
                "text": gold_spans if gold_spans else [""],
                "answer_start": [-1] * len(gold_spans) # answer_start is often needed, -1 if not applicable
            }
        })

    if not formatted_predictions: # No valid data to score
        return {"f1": 0.0, "exact_match": 0.0}

    results = squad_metric.compute(predictions=formatted_predictions, references=formatted_references)
    
    return {"f1": results.get("f1", 0.0) / 100.0, "exact_match": results.get("exact_match", 0.0) / 100.0}


def compute_f1(predictions: List[str], references: List[str], *args, **kwargs) -> float:
    """
    Computes the F1 score for the DROP dataset (multi-span, multi-reference).
    Args:
        predictions (List[str]): Each is a JSON string decoding to a list of predicted answer spans.
        references (List[str]): Each is a JSON string decoding to an object with "spans" (list of gold answer spans).
    Returns:
        float: F1 score (0-1)
    """
    results = compute_drop_f1_em_metric(predictions, references)
    return results.get("f1", 0.0)

def compute_exact_match(predictions: List[str], references: List[str], *args, **kwargs) -> float:
    """
    Computes the Exact Match (EM) score for the DROP dataset (multi-span, multi-reference).
    Args:
        predictions (List[str]): Each is a JSON string decoding to a list of predicted answer spans.
        references (List[str]): Each is a JSON string decoding to an object with "spans" (list of gold answer spans).
    Returns:
        float: Exact Match score (0-1)
    """
    results = compute_drop_f1_em_metric(predictions, references)
    return results.get("exact_match", 0.0)

def load(make_prompt: bool, *args, **kwargs) -> Tuple[
    Union[List[Tuple[str, str]], List[str]], 
    List[str], 
    Union[List[Tuple[str, str]], List[str]], 
    List[str]
]:
    """
    Loads the DROP dataset from pre-processed CSV files.

    The CSVs are expected to have 'passage', 'question', and 'answers_spans' columns.
    'answers_spans' should contain a JSON string like:
    '{"spans": ["answer span 1", "answer span 2"], "types": ["type1", "type2"]}'

    Args:
        make_prompt (bool): If True, X features are converted into prompt strings.
                            Otherwise, X is a list of (passage, question) tuples.

    Returns:
        Tuple containing (X_train, y_train, X_test, y_test).
        - X_train/X_test: List of (passage, question) tuples or prompt strings.
        - y_train/y_test: List of strings, where each string is a JSON representation
                          of an object containing answer spans and their types 
                          (e.g., '{"spans": ["span1", "span2"], "types": ["typeA", "typeB"]}').
    """
    try:
        # This assumes `download("drop")` makes train.csv and test.csv available
        # in the directory structure that `datapath("drop/drop")` points to.
        # Typically, `datapath("drop")` would give the base directory for "drop",
        # and then you might have subdirectories or files.
        # The original script used `datapath("drop/drop")`. We'll stick to it.
        download("drop") 
    except Exception as e:
        print(
            "Error during autogoal.datasets.download('drop'). "
            "This may be caused by a bad connection or misconfigured dataset path. "
            "Please ensure 'train.csv' and 'test.csv' for DROP are correctly placed."
        )
        raise e

    data_folder_path = datapath("drop/drop")
    
    X_train: Union[List[Tuple[str, str]], List[str]] = []
    y_train: List[str] = []
    X_test: Union[List[Tuple[str, str]], List[str]] = []
    y_test: List[str] = []
    
    # Helper function to read CSV files for QA

    def read_csv_drop(file_path: str) -> Tuple[List[Tuple[str, str]], List[str]]:
        X = []
        y = []
        try:
            with open(file_path, "r", encoding="utf-8") as fd:
                reader = csv.DictReader(fd)
                for i, row in enumerate(reader):
                    passage = row.get("passage", "")
                    question = row.get("question", "")
                    answers_json_str = row.get("answers_spans", "")

                    if passage and question and answers_json_str:
                        try:
                            answers_data = json.loads(answers_json_str)
                            spans_list = answers_data.get("spans")
                            types_list = answers_data.get("types") # Get types

                            # Ensure all components are valid and lists of the same length
                            if isinstance(spans_list, list) and \
                               isinstance(types_list, list) and \
                               len(spans_list) == len(types_list):
                                # Convert all spans to string, just in case they aren't
                                cleaned_spans = [str(s) for s in spans_list]
                                # Only store the spans list as the answer (less noise for tuning)
                                y_target_string = json.dumps(cleaned_spans)
                                X.append((passage, question))
                                y.append(y_target_string)
                            else:
                                print(f"Warning: Row {i+1} in {file_path}: 'spans' and/or 'types' in '{answers_json_str}' are missing, not lists, or have mismatched lengths. Skipping.")
                        except json.JSONDecodeError:
                            print(f"Warning: Row {i+1} in {file_path}: Could not parse JSON from 'answers_spans' field: '{answers_json_str}'. Skipping.")
                        except Exception as e:
                            print(f"Warning: Row {i+1} in {file_path}: An unexpected error occurred while processing answers_spans: {e}. Skipping.")
                    else:
                        # Handle missing essential fields
                        missing_fields = []
                        if not passage: missing_fields.append("'passage'")
                        if not question: missing_fields.append("'question'")
                        if not answers_json_str: missing_fields.append("'answers_spans'")
                        print(f"Warning: Row {i+1} in {file_path}: Missing required field(s): {', '.join(missing_fields)}. Skipping.")
            if not X:
                print(f"Warning: No valid data loaded from {file_path}. Please check the file format and content.")

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}. Please ensure the CSV files are correctly placed.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while reading {file_path}: {e}")
            raise
        return X, y

     # Read training and testing data
    train_file_path = str(data_folder_path / "train.csv")
    test_file_path = str(data_folder_path / "test.csv")

    print(f"Attempting to load training data from: {train_file_path}")
    X_train_tuples, y_train = read_csv_drop(train_file_path)
    print(f"Attempting to load test data from: {test_file_path}")
    X_test_tuples, y_test = read_csv_drop(test_file_path)

    if make_prompt:
        X_train = [f"Passage: {passage}\nQuestion: {question}\nAnswer:" for passage, question in X_train_tuples]
        X_test = [f"Passage: {passage}\nQuestion: {question}\nAnswer:" for passage, question in X_test_tuples]
    else:
        X_train = X_train_tuples
        X_test = X_test_tuples
        
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    print("DROP dataset loading and new metric functions are defined.")
    print("Run tests separately if you have them for the new evaluate-based metrics.")
    pass