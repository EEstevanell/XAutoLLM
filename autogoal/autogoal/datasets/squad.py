import csv
from typing import Dict, Tuple, List, Optional

from autogoal.datasets import download, datapath


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
        raise ValueError("SQuAD metric not loaded. Cannot compute F1 score.")

    try:
        formatted_data = _format_squad_inputs(reference_texts, prediction_texts)
        results = squad_metric.compute(
            predictions=formatted_data["predictions"],
            references=formatted_data["references"],
        )
        print(f"F1 Score: {results['f1']}")
        return results["f1"]
    except Exception as e:
        print(f"Error during F1 computation: {e}")
        raise e  # Raise the exception to indicate failure

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
        raise ValueError("SQuAD metric not loaded. Cannot compute Exact Match score.")

    try:
        formatted_data = _format_squad_inputs(reference_texts, prediction_texts)
        results = squad_metric.compute(
            predictions=formatted_data["predictions"],
            references=formatted_data["references"],
        )
        print(f"Exact Match Score: {results['exact_match']}")
        return results["exact_match"]
    except Exception as e:
        print(f"Error during Exact Match computation: {e}")
        raise e

def load(make_prompt: bool, *args, **kwargs):
    try:
        download("squad")
    except Exception as e:
        print(
            "Error loading data. This may be caused due to bad connection. "
            "Please delete badly downloaded data and retry."
        )
        raise e

    path = datapath("squad/squad")

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # Helper function to read CSV files for QA: X=(context, question), y=answer_text
    def read_csv(file_path: str) -> Tuple[List[Tuple[str, str]], List[str]]:
        X = []
        y = []
        with open(file_path, "r", encoding="utf-8") as fd:
            reader = csv.DictReader(fd)
            for row in reader:
                # Ensure required fields exist and are not empty
                context = row.get("context", "")
                question = row.get("question", "")
                answer = row.get("answer_text", "")
                if context and question and answer:
                    X.append((context, question))
                    y.append(answer)
        return X, y

    # Read training and testing data
    X_train, y_train = read_csv(str(path / "train.csv"))
    X_test, y_test = read_csv(str(path / "test.csv"))

    if (make_prompt):
        # Convert to prompt format: (context, question) -> answer_text
        X_train = [f"Context: {context}\nQuestion: {question}\nAnswer:" for context, question in X_train]
        X_test = [f"Context: {context}\nQuestion: {question}\nAnswer:" for context, question in X_test]

    return X_train, y_train, X_test, y_test

def test_squad_metrics():
    """Unit tests for compute_squad_f1 and compute_squad_exact_match covering key cases."""
    # Helper for reference and prediction lists
    def ref(pred):
        return [pred]

    # 1. Perfect match
    preds = ["a test answer"]
    refs = ["a test answer"]
    f1 = compute_squad_f1(refs, preds)
    em = compute_squad_exact_match(refs, preds)
    assert f1 == 100.0 and em == 100.0, f"Failed perfect match: f1={f1}, em={em}"

    # 2. Case/whitespace/punctuation difference (should normalize)
    preds = ["A test, answer!"]
    refs = ["a test answer"]
    f1 = compute_squad_f1(refs, preds)
    em = compute_squad_exact_match(refs, preds)
    assert f1 == 100.0 and em == 100.0, f"Failed normalization: f1={f1}, em={em}"

    # 3. Partial match
    preds = ["a test"]
    refs = ["a test answer"]
    f1 = compute_squad_f1(refs, preds)
    em = compute_squad_exact_match(refs, preds)
    assert 0 < f1 < 100.0 and em == 0.0, f"Failed partial match: f1={f1}, em={em}"

    # 4. No match
    preds = ["foo"]
    refs = ["bar"]
    f1 = compute_squad_f1(refs, preds)
    em = compute_squad_exact_match(refs, preds)
    assert f1 == 0.0 and em == 0.0, f"Failed no match: f1={f1}, em={em}"

    # 5. Empty prediction and gold
    preds = [""]
    refs = [""]
    f1 = compute_squad_f1(refs, preds)
    em = compute_squad_exact_match(refs, preds)
    assert f1 == 0.0 and em == 100.0, f"Failed empty match: f1={f1}, em={em}"

    # 6. Empty prediction, non-empty gold
    preds = [""]
    refs = ["foo"]
    f1 = compute_squad_f1(refs, preds)
    em = compute_squad_exact_match(refs, preds)
    assert f1 == 0.0 and em == 0.0, f"Failed empty pred: f1={f1}, em={em}"

    # 7. Malformed input (mismatched lengths)
    try:
        compute_squad_f1(["foo", "bar"], ["foo"])
    except ValueError:
        pass
    else:
        assert False, "Failed to raise ValueError for mismatched input lengths"

    print("All SQuAD metric unit tests passed.")

if __name__ == "__main__":
    print("Running SQuAD metric unit tests...")
    test_squad_metrics()
    # Example usage with ordinal encoding
    X_train, y_train, X_test, y_test = load(make_prompt=True)
    print("Training (context,question) len):", len(X_train))
    print("Training (context,question) 1st):", X_train[1])
    print("Training answers 1st:", y_train[:3])
    print("Testing (context,question) len:", len(X_test))
    print("Testing (context,question)  1st:", X_test[1])
    print("Testing answers 1st:", y_test[:3])

    # --- Dummy test for SQuAD metric ---
    if len(y_test) >= 3:
        print("\n--- Testing SQuAD Metric (F1 & EM) using huggingface/evaluate ---")
        # 1. Perfect match
        # 2. Completely wrong
        # 3. Partial overlap
        dummy_predictions = [
            y_test[0],  # perfect match
            "Wrong Answer",  # completely wrong
            y_test[2].split(",")[0]  # partial match (e.g., "Santa Clara" vs "Santa Clara, California")
        ]
        dummy_references = y_test[:3]

        f1 = compute_squad_f1(dummy_references, dummy_predictions)
        em = compute_squad_exact_match(dummy_references, dummy_predictions)
        print(f"Dummy SQuAD F1: {f1}")
        print(f"Dummy SQuAD EM: {em}")
    else:
        print("Not enough test data to run SQuAD metric dummy test.")
