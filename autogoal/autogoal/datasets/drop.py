from collections import Counter
import csv
import json
import re
import string
from typing import Dict, Tuple, List, Optional, Union

from autogoal.datasets import download, datapath

# --- DROP Official Metric Implementation (AllenNLP style, multi-span, multi-reference) ---
def _normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_punc(remove_articles(lower(s))))

def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def _multi_span_metric(prediction_spans: List[str], gold_spans: List[str]) -> Tuple[float, float]:
    """
    Compute F1 and EM for multi-span answers as in the official DROP evaluation.
    """
    # If both are empty, it's a perfect match
    if not prediction_spans and not gold_spans:
        return 1.0, 1.0
    # If only one is empty, EM=F1=0
    if not prediction_spans or not gold_spans:
        return 0.0, 0.0
    # For EM: unordered set match
    if set(map(_normalize_answer, prediction_spans)) == set(map(_normalize_answer, gold_spans)):
        em = 1.0
    else:
        em = 0.0
    # For F1: compute all pairwise F1s, find best alignment
    pred_used = set()
    gold_used = set()
    f1s = []
    for i, pred in enumerate(prediction_spans):
        best_f1 = 0.0
        best_j = -1
        for j, gold in enumerate(gold_spans):
            if j in gold_used:
                continue
            f1 = _f1_score(pred, gold)
            if f1 > best_f1:
                best_f1 = f1
                best_j = j
        if best_j >= 0:
            pred_used.add(i)
            gold_used.add(best_j)
            f1s.append(best_f1)
    # Unmatched predictions/golds get F1=0
    f1s.extend([0.0] * (len(prediction_spans) - len(pred_used)))
    f1s.extend([0.0] * (len(gold_spans) - len(gold_used)))
    mean_f1 = sum(f1s) / max(len(prediction_spans), len(gold_spans))
    return mean_f1, em

def compute_drop_f1_em_metric(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes the F1 and Exact Match (EM) scores for the DROP dataset (multi-span, multi-reference).
    Args:
        predictions (List[str]): Each is a JSON string decoding to a list of predicted answer spans.
        references (List[str]): Each is a JSON string decoding to an object with "spans" (list of gold answer spans).
    Returns:
        Dict[str, float]: {"f1": ..., "exact_match": ...}
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    total_f1 = 0.0
    total_em = 0.0
    count = 0
    for pred_json_str, ref_json_str in zip(predictions, references):
        # Parse prediction
        try:
            pred_spans = json.loads(pred_json_str)
            if not isinstance(pred_spans, list):
                pred_spans = [str(pred_spans)]
            pred_spans = [str(s) for s in pred_spans]
        except Exception:
            pred_spans = []
        # Parse reference
        try:
            ref_obj = json.loads(ref_json_str)
            gold_spans = ref_obj.get("spans", [])
            if not isinstance(gold_spans, list):
                gold_spans = [str(gold_spans)]
            gold_spans = [str(s) for s in gold_spans]
        except Exception:
            gold_spans = []
        f1, em = _multi_span_metric(pred_spans, gold_spans)
        total_f1 += f1
        total_em += em
        count += 1
    if count == 0:
        return {"f1": 0.0, "exact_match": 0.0}
    return {"f1": total_f1 / count, "exact_match": total_em / count}

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
    """
    Computes the F1 and Exact Match (EM) scores for the DROP dataset (multi-span, multi-reference).
    Args:
        predictions (List[str]): Each is a JSON string decoding to a list of predicted answer spans.
        references (List[str]): Each is a JSON string decoding to an object with "spans" (list of gold answer spans).
    Returns:
        Dict[str, float]: {"f1": ..., "exact_match": ...}
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    total_f1 = 0.0
    total_em = 0.0
    count = 0
    for pred_json_str, ref_json_str in zip(predictions, references):
        # Parse prediction
        try:
            pred_spans = json.loads(pred_json_str)
            if not isinstance(pred_spans, list):
                pred_spans = [str(pred_spans)]
            pred_spans = [str(s) for s in pred_spans]
        except Exception:
            pred_spans = []
        # Parse reference
        try:
            ref_obj = json.loads(ref_json_str)
            gold_spans = ref_obj.get("spans", [])
            if not isinstance(gold_spans, list):
                gold_spans = [str(gold_spans)]
            gold_spans = [str(s) for s in gold_spans]
        except Exception:
            gold_spans = []
        f1, em = _multi_span_metric(pred_spans, gold_spans)
        total_f1 += f1
        total_em += em
        count += 1
    if count == 0:
        return {"f1": 0.0, "exact_match": 0.0}
    return {"f1": total_f1 / count, "exact_match": total_em / count}

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
                                
                                # Convert all spans and types to string, just in case they aren't
                                cleaned_spans = [str(s) for s in spans_list]
                                cleaned_types = [str(t) for t in types_list]

                                # Store the full structure including types for y
                                y_target_obj = {"spans": cleaned_spans, "types": cleaned_types}
                                y_target_string = json.dumps(y_target_obj)
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

def test_drop_metrics():
    """Unit tests for compute_drop_f1_em_metric covering key cases."""
    # Helper to wrap spans as json for predictions and references
    def pred_json(spans):
        return json.dumps(spans)
    def ref_json(spans):
        return json.dumps({"spans": spans, "types": ["span"] * len(spans)})

    # 1. Perfect match, single span
    preds = [pred_json(["a test answer"])]
    refs = [ref_json(["a test answer"])]
    result = compute_drop_f1_em_metric(preds, refs)
    assert result["f1"] == 1.0 and result["exact_match"] == 1.0, f"Failed perfect match: {result}"

    # 2. Case/whitespace/punctuation difference (should normalize)
    preds = [pred_json(["A test, answer!"])]
    refs = [ref_json(["a test answer"])]
    result = compute_drop_f1_em_metric(preds, refs)
    assert result["f1"] == 1.0 and result["exact_match"] == 1.0, f"Failed normalization: {result}"

    # 3. Partial match
    preds = [pred_json(["a test"])]
    refs = [ref_json(["a test answer"])]
    result = compute_drop_f1_em_metric(preds, refs)
    assert 0 < result["f1"] < 1.0 and result["exact_match"] == 0.0, f"Failed partial match: {result}"

    # 4. No match
    preds = [pred_json(["foo"])]
    refs = [ref_json(["bar"])]
    result = compute_drop_f1_em_metric(preds, refs)
    assert result["f1"] == 0.0 and result["exact_match"] == 0.0, f"Failed no match: {result}"

    # 5. Multi-span, unordered, perfect match
    preds = [pred_json(["one", "two"])]
    refs = [ref_json(["two", "one"])]
    result = compute_drop_f1_em_metric(preds, refs)
    assert result["f1"] == 1.0 and result["exact_match"] == 1.0, f"Failed multi-span unordered: {result}"

    # 6. Multi-span, partial match
    preds = [pred_json(["one", "three"])]
    refs = [ref_json(["one", "two"])]
    result = compute_drop_f1_em_metric(preds, refs)
    assert 0 < result["f1"] < 1.0 and result["exact_match"] == 0.0, f"Failed multi-span partial: {result}"

    # 7. Empty prediction and gold
    preds = [pred_json([])]
    refs = [ref_json([])]
    result = compute_drop_f1_em_metric(preds, refs)
    assert result["f1"] == 1.0 and result["exact_match"] == 1.0, f"Failed empty match: {result}"

    # 8. Empty prediction, non-empty gold
    preds = [pred_json([])]
    refs = [ref_json(["foo"])]
    result = compute_drop_f1_em_metric(preds, refs)
    assert result["f1"] == 0.0 and result["exact_match"] == 0.0, f"Failed empty pred: {result}"

    # 9. Malformed prediction JSON
    preds = ['[not valid json']
    refs = [ref_json(["foo"])]
    result = compute_drop_f1_em_metric(preds, refs)
    assert result["f1"] == 0.0 and result["exact_match"] == 0.0, f"Failed malformed pred: {result}"

    # 10. Malformed reference JSON
    preds = [pred_json(["foo"])]
    refs = ['{not valid json']
    result = compute_drop_f1_em_metric(preds, refs)
    assert result["f1"] == 0.0 and result["exact_match"] == 0.0, f"Failed malformed ref: {result}"

    print("All DROP metric unit tests passed.")


if __name__ == "__main__":
    print("Running DROP metric unit tests...")
    test_drop_metrics()
    print("\nAttempting to load data with make_prompt=True...")
    try:
        X_train_data, y_train_data, X_test_data, y_test_data = load(make_prompt=True)
        print("\nSample of loaded data:")
        print("Number of training examples loaded:", len(X_train_data))
        if X_train_data and y_train_data:
            print("First training prompt:", X_train_data[0])
            print("First training answer string (target for LLM):", y_train_data[0])
        print("\nNumber of testing examples loaded:", len(X_test_data))
        if X_test_data and y_test_data:
            print("First testing prompt:", X_test_data[0])
            print("First testing answer string (reference for metric):", y_test_data[0])
        # Example usage of the F1 metric
        if X_test_data and y_test_data and len(y_test_data) >=3 :
            print("\n--- Testing DROP Metric (F1 & EM) using huggingface/evaluate ---")
            pred_spans_for_y0 = []
            if len(y_test_data) > 0:
                try:
                    data_y0 = json.loads(y_test_data[0])
                    if isinstance(data_y0, dict) and "spans" in data_y0 and isinstance(data_y0["spans"], list):
                        pred_spans_for_y0 = [str(s) for s in data_y0["spans"]]
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse y_test_data[0] for dummy predictions: {y_test_data[0]}")
            pred_first_span_from_y1 = "single pred"
            if len(y_test_data) > 1:
                try:
                    data_y1 = json.loads(y_test_data[1])
                    if isinstance(data_y1, dict) and "spans" in data_y1 and isinstance(data_y1["spans"], list) and data_y1["spans"]:
                        pred_first_span_from_y1 = str(data_y1["spans"][0])
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse y_test_data[1] for dummy predictions: {y_test_data[1]}")
            pred_spans_from_y2 = []
            if len(y_test_data) > 2:
                try:
                    data_y2 = json.loads(y_test_data[2])
                    if isinstance(data_y2, dict) and "spans" in data_y2 and isinstance(data_y2["spans"], list):
                        pred_spans_from_y2 = [str(s) for s in data_y2["spans"]]
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse y_test_data[2] for dummy predictions: {y_test_data[2]}")
            dummy_predictions = [
                json.dumps(pred_spans_for_y0),
                json.dumps(["a different span", "another one completely"]),
                '["this is not quite json',
                json.dumps([pred_first_span_from_y1]),
                json.dumps(pred_spans_from_y2 + ["an extra predicted span"])
            ]
            num_metric_samples = min(len(dummy_predictions), len(y_test_data))
            if num_metric_samples > 0:
                metric_predictions = dummy_predictions[:num_metric_samples]
                metric_references = y_test_data[:num_metric_samples]
                print(f"Calculating F1 & EM for {num_metric_samples} dummy prediction(s):")
                for i in range(num_metric_samples):
                    print(f"  Pred {i+1}: {metric_predictions[i]}")
                    print(f"  Ref  {i+1}: {metric_references[i]}")
                metric_results = compute_drop_f1_em_metric(metric_predictions, metric_references)
                print(f"Computed DROP metrics: {metric_results}")
            else:
                print("Not enough test data to run DROP metric examples.")
        else:
            print("Skipping metric example due to insufficient test data (need at least 3 samples).")
    except Exception as e:
        print(f"An error occurred during the load or metric example: {e}")