import csv
from typing import Tuple, List, Optional

from autogoal.datasets import download, datapath

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


if __name__ == "__main__":
    # Example usage with ordinal encoding
    X_train, y_train, X_test, y_test = load(make_prompt=True)
    print("Training (context,question) len):", len(X_train))
    print("Training (context,question) 1st):", X_train[1])
    print("Training answers 1st:", y_train[:51])
    print("Testing (context,question) len:", len(X_test))
    print("Testing (context,question)  1st:", X_test[1])
    print("Testing answers 1st:", y_test[:1])
